import os
import random
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import typing as tp

from utils.config import Config
from utils.logger import summarize
from utils.script_util import save_checkpoint
from utils.curriculum_scheduler import CurriculumScheduler
from jen1.model.model import UNetCFG1d
from jen1.diffusion.gdm.gdm import GaussianDiffusion
from jen1.conditioners import MultiConditioner

class UnifiedMultiTaskTrainer(nn.Module):
    def __init__(self,
                 config: Config,
                 rank: int,
                 epoch_str: int,
                 global_step: int, 
                 model: UNetCFG1d,
                 diffusion: tp.Optional[GaussianDiffusion],
                 conditioner: MultiConditioner,
                 dls,
                 optimizer,
                 lr_scheduler,
                 scaler,
                 logger,
                 writers,
                 grad_clip,
                 grad_accum_every,
                 cross_attn_cond_ids=['prompt'],
                 global_cond_ids= [],
                 input_concat_ids= ['masked_input', 'mask'],
                 curriculum_scheduler: CurriculumScheduler = None
                 ):
        """
        Args:
            config: Configuration object containing settings for the training process.
            rank: Rank in distributed training, indicating the specific process in a multi-GPU setup.
            epoch_str: Starting epoch number for the training process.
            global_step: Global step number, used to track overall progress in training.
            model: The UNet model, customized for 1-dimensional data processing.
            diffusion: Optional diffusion model for generating data.
            conditioner: Module for conditioning the input, used when specific conditions are applied.
            dls: Tuple containing data loaders for training (train_dl) and validation (valid_dl).
            optimizer: The optimizer used for training the model.
            lr_scheduler: Learning rate scheduler for adjusting the learning rate during training.
            scaler: Scaler for gradient scaling, typically used in mixed precision training.
            logger: Logger for tracking the training process and recording metrics.
            writers: Tuple of writers for writing logs, separated for training and validation.
            grad_clip: Maximum value for gradient clipping to prevent exploding gradients.
            grad_accum_every: Specifies how often to accumulate gradients before updating weights.
            cross_attn_cond_ids: Keys in the output dictionary from the conditioner, used for cross-attention.
            global_cond_ids: Keys in the output dictionary from the conditioner, used for global conditions.
            input_concat_ids: Keys in the output dictionary from the conditioner, whose outputs are concatenated channel-wise to the model input.
            curriculum_scheduler: Scheduler for curriculum training, to progressively increase the difficulty of the training data.
        """
        super().__init__()
        self.config=config
        self.tasks = self.config.tasks
        self.rank = rank
        self.epoch_str = epoch_str
        self.global_step = global_step
        self.grad_clip = grad_clip
        self.grad_accum_every = grad_accum_every
        self.model = model
        self.diffusion = diffusion
        self.conditioner = conditioner
        self.train_dl, self.valid_dl = dls        
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.logger = logger
        self.writer, self.writer_val = writers
        
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        
        self.best_avg_total_loss = float('inf')
        self.curriculum_scheduler = curriculum_scheduler
        
    def curriculum_train(self):
        assert self.curriculum_scheduler is not None, "Curriculum scheduler is not provided"
        num_curriculum_stages = self.curriculum_scheduler.curriculum_stages
        
        for stage in range(self.curriculum_scheduler.current_stage, num_curriculum_stages + 1):
            start_epoch, end_epoch = self.curriculum_scheduler.get_current_stage_epochs()
            self.train_loop(self.epoch_str + start_epoch, self.epoch_str + end_epoch)
            self.curriculum_scheduler.update_stage()
        
    def eval_all_tasks(self, epoch):
        avg_total_loss = 0
        
        all_task_loss_dict, task_count = self.eval()
        for task in self.tasks:
            avg_loss = all_task_loss_dict[task] / task_count if task_count > 0 else 0
            avg_total_loss += avg_loss
            self.logger.info(f'Average validation loss for task {task}: {avg_loss}')
            if self.rank == 0:
                scalars = {f'loss/val_{task}': avg_loss}
                summarize(writer=self.writer, global_step=self.global_step, scalars=scalars)
        
        self.logger.info(f'Average total validation loss: {avg_total_loss}')
        if avg_total_loss < self.best_avg_total_loss:
            self.best_avg_total_loss = avg_total_loss
            self.logger.info(f'New best average total validation loss: {self.best_avg_total_loss}')
            save_checkpoint(model=self.model, optimizer=self.optimizer,
                                        lr=self.config.optimizer_config.lr, iteration=epoch,
                                        checkpoint_path=os.path.join(self.config.save_dir, f'Jen1_step_{self.global_step}_loss_{self.best_avg_total_loss}.pth'),
                                        logger=self.logger)
        if self.rank == 0:
            scalars = {'loss/val_total': avg_total_loss}
            summarize(writer=self.writer, global_step=self.global_step, scalars=scalars)
        
        self.model.train()
    
    def eval(self):
        self.model.eval()
        count = 0
        loss_dict = {task: 0 for task in self.tasks}
        with torch.no_grad():
            for batch_idx, (audio_emb, metadata, demix_embs_dict) in enumerate(self.valid_dl):
                b, _, _, device = *audio_emb.shape, self.config.device
                assert b % len(self.tasks) == 0, "Batch size must be divisible by the number of tasks"
                sub_batch_size = b // len(self.tasks)
                
                for i, task in enumerate(self.tasks):
                    start_idx = i * sub_batch_size
                    end_idx = start_idx + sub_batch_size
                    sub_audio_emb = audio_emb[start_idx:end_idx]
                    sub_metadata = metadata[start_idx:end_idx]
                    sub_bass_embs = demix_embs_dict['bass'][start_idx:end_idx]
                    sub_drums_embs = demix_embs_dict['drums'][start_idx:end_idx]
                    sub_other_embs = demix_embs_dict['other'][start_idx:end_idx]
                    sub_demix_embs_dict = {'bass': sub_bass_embs,
                                           'drums': sub_drums_embs,
                                           'other': sub_other_embs}
                    b, c, _, device = *sub_audio_emb.shape, self.config.device
                    num_tracks = len(sub_demix_embs_dict)
                    
                    current_stage = self.curriculum_scheduler.current_stage
                    selected_audio_emb, remaining_audio_emb, selected_keys, remaining_keys = \
                        self.select_random_tracks(sub_demix_embs_dict, current_stage, num_tracks)
                    prefix_prompt = self.create_prefix_prompt(selected_keys)
                    for item in sub_metadata:
                        item['prompt'] = prefix_prompt + '' + item['prompt']
                    masked_input, mask, causal = self.random_mask(sub_audio_emb, sub_audio_emb.shape[2], task)
                    conditioning = self.conditioner(sub_metadata, self.config.device)
                    conditioning['masked_input'] = masked_input
                    conditioning['mask'] = mask
                    conditioning = self.get_conditioning(conditioning)
                    
                    if self.config.diffusion_type == 'gdm':
                        num_timesteps = self.diffusion.num_timesteps
                        t_i = torch.randint(1, num_timesteps-2, (b,), device=device).long()
                        t_for_cond = torch.zeros(b, dtype=torch.long, device=device)
                        for i in range(b):
                            t_for_cond[i] = random.choice([0, t_i[i], num_timesteps-1])
                        with autocast(enabled=self.config.use_fp16):
                            loss = self.diffusion.training_losses(self.model, (selected_audio_emb, remaining_audio_emb),
                                                                   (t_i, t_for_cond), conditioning, causal=causal)
                    loss_dict[task] += loss.item()
                count += 1
                                
        return loss_dict, count
        
    def train_loop(self, str_epoch, end_epoch):
        grad_accum = 0
        all_loss = 0
        loss_dict = {task: 0 for task in self.tasks}
        
        for epoch in range(str_epoch, int(end_epoch + 1)):
            for batch_idx, (audio_emb, metadata, demix_embs_dict) in enumerate(self.train_dl):
                all_task_loss, all_loss_dict = self.train(audio_emb=audio_emb, metadata=metadata, demix_embs_dict=demix_embs_dict)
                all_loss += all_task_loss.item() / self.grad_accum_every
                for task in self.tasks:
                    loss_dict[task] += (all_loss_dict[task] / self.grad_accum_every)
                
                if grad_accum == 0:
                    self.optimizer.zero_grad()
                self.scaler.scale(all_task_loss / self.grad_accum_every).backward()
                grad_accum += 1
                
                if grad_accum == self.grad_accum_every:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.lr_scheduler.step()
                    self.scaler.update()
                    grad_accum = 0
                
                    if self.rank == 0:
                        loss_text_guided = loss_dict['text_guided']
                        loss_inpaint = loss_dict['music_inpaint']
                        loss_cont = loss_dict['music_cont']
                        if self.global_step % self.config.log_interval == 0:
                            lr = self.optimizer.param_groups[0]['lr']
                            self.logger.info('Train Epoch: {}, [{:.0f}%]'.format(
                                epoch, 100. * batch_idx / len(self.train_dl)
                                ))
                            self.logger.info(
                                f'loss: {all_loss} '
                                f'loss_text_guided: {loss_text_guided} '
                                f'loss_inpaint: {loss_inpaint} '
                                f'loss_cont: {loss_cont} '
                                f'global_step: {self.global_step}, lr:{lr}')
                            scalars = {'loss/train': all_loss,
                                    'loss_text_guided/train': loss_text_guided,
                                    'loss_inpaint/train': loss_inpaint,
                                    'loss_cont/train': loss_cont}
                            summarize(writer=self.writer, global_step=self.global_step, scalars=scalars)
                            
                    loss_dict = {task: 0 for task in self.tasks}
                    all_loss = 0
                    
                if self.global_step % self.config.eval_interval == 0:
                    self.eval_all_tasks(epoch=epoch)
                
                self.global_step += 1   
    
    def train(self, audio_emb, metadata, demix_embs_dict):
        '''
        demix_embs_dict contains the latent representations for each track:
            demix_embs_dict = {'bass': data_for_bass, 
                            'drums': data_for_drum,
                            'other': data_for_other }
        The shape of each latent representation is [b, c, t], with c = 128.
        '''
        loss_dict = {task: 0 for task in self.tasks}
        all_loss = torch.tensor(0.0, device=self.config.device)
        batch_size = audio_emb.size(0)
        num_tracks = len(demix_embs_dict)
        assert batch_size % len(self.tasks) == 0, "Batch size must be divisible by the number of tasks"
        # This part evenly distributes samples in the batch among the tasks ('text_guided', 'music_inpaint', 'music_cont').
        # Therefore, the batch must be divisible by the number of tasks.
        sub_batch_size = batch_size // len(self.tasks)
        for i, task in enumerate(self.tasks):
            start_idx = i * sub_batch_size
            end_idx = start_idx + sub_batch_size
            sub_audio_emb = audio_emb[start_idx:end_idx]
            sub_metadata = metadata[start_idx:end_idx]
            sub_bass_embs = demix_embs_dict['bass'][start_idx:end_idx]
            sub_drums_embs = demix_embs_dict['drums'][start_idx:end_idx]
            sub_other_embs = demix_embs_dict['other'][start_idx:end_idx]
            sub_demix_embs_dict = {'bass': sub_bass_embs,
                                   'drums': sub_drums_embs,
                                   'other': sub_other_embs}
            assert num_tracks > 1, 'num_tracks must be greater than 1'
            self.model.train()
            b, _, _, device = *sub_audio_emb.shape, self.config.device

            # Retrieve the current curriculum stage; the number in current_stage represents the number of tracks to generate.
            # For more details, see section 4.3 PROGRESSIVE CURRICULUM TRAINING STRATEGY in the JEN-1-Composer paper (https://arxiv.org/abs/2310.19180).
            current_stage = self.curriculum_scheduler.current_stage 
            # selected_audio_emb are the tracks selected for the current curriculum stage;
            # remaining_audio_emb are the tracks not selected for the current stage.
            # selected_keys and remaining_keys represent the names of these tracks.
            selected_audio_emb, remaining_audio_emb, selected_keys, remaining_keys = \
                self.select_random_tracks(sub_demix_embs_dict, current_stage, num_tracks)
            prefix_prompt = self.create_prefix_prompt(selected_keys)
            for item in metadata:
                # If you know whether to add prefix prompts or prefix tuning, please fix this part.
                item['prompt'] = prefix_prompt + ' ' + item['prompt']
            # The masked_input in this part is the audio_emb masked according to each task;
            # mask is the mask corresponding to each task, and causal is the mode for each task.
            # This follows the omnidirectional latent diffusion model from JEN-1.
            # masked_input has 128 channels, mask has 1 channel; in total, 129 channels are concatenated channel-wise as the model input.
            # For more details, see section 4.2 OMNIDIRECTIONAL LATENT DIFFUSION MODELS in the JEN-1 paper (https://arxiv.org/abs/2308.04729).
            masked_input, mask, causal = self.random_mask(sub_audio_emb, sub_audio_emb.shape[2], task)
            conditioning = self.conditioner(sub_metadata, self.config.device)
            conditioning['masked_input'] = masked_input
            conditioning['mask'] = mask
            conditioning = self.get_conditioning(conditioning)

            if self.config.diffusion_type == 'gdm':
                num_timesteps = self.diffusion.num_timesteps
                # Selected tracks choose timesteps from 1 to T-1.
                # For the remaining tracks, choose from [0, t_i, T].
                # 0 is for Conditional Generation, t_i for Joint Generation, and T for Marginal Generation.
                # For more details, see section 5.1 SETUP-Implementation Details and Figure 2 in the JEN-1-Composer paper (https://arxiv.org/abs/2310.19180).
                t_i = torch.randint(1, num_timesteps-2, (b,), device=device).long()
                t_for_cond = torch.zeros(b, dtype=torch.long, device=device)
                for i in range(b):
                    t_for_cond[i] = random.choice([0, t_i[i], num_timesteps-1])
                with autocast(enabled=self.config.use_fp16):
                    loss = self.diffusion.training_losses(self.model, (selected_audio_emb, remaining_audio_emb), 
                                                    (t_i, t_for_cond), conditioning, causal=causal)
                    print('loss:', loss, 'task:', task)
            loss_dict[task] += loss.item()
            all_loss += loss
        
        return all_loss, loss_dict
    
    def random_mask(self, sequence, max_mask_length, task):
        b, _, sequence_length = sequence.size()
        
        masks = []
        if task.lower() == 'text_guided':
            mask = torch.zeros((1, 1, sequence_length)).to(sequence.device)
            masks.append(mask)
            causal = random.choices([True, False])[0]
        elif task.lower() == 'music_inpaint':
            mask_length = random.randint(sequence_length*0.2, sequence_length*0.8)
            mask_start = random.randint(0, sequence_length-mask_length)
            
            mask = torch.ones((1, 1, sequence_length))
            mask[:, :, mask_start:mask_start+mask_length] = 0
            mask = mask.to(sequence.device)
            
            masks.append(mask)
            causal = False
        elif task.lower() == 'music_cont':
            mask_length = random.randint(sequence_length*0.2, sequence_length*0.8)
            
            mask = torch.ones((1, 1, sequence_length))
            mask[:, :, -mask_length:] = 0
            masks.append(mask)
            causal = True
        
        masks = masks * b
        mask = torch.cat(masks, dim=0).to(sequence.device)
        
        masked_sequence = sequence * mask
        
        return masked_sequence, mask, causal
    
    def get_conditioning(self, cond):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = torch.cat([cond[key][0] for key in self.cross_attn_cond_ids], dim=1)
            cross_attention_masks = torch.cat([cond[key][1] for key in self.cross_attn_cond_ids], dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_cond = torch.cat([cond[key][0] for key in self.global_cond_ids], dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)
        
        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([cond[key] for key in self.input_concat_ids], dim=1)

        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_masks": cross_attention_masks,
            "global_cond": global_cond,
            "input_concat_cond": input_concat_cond
        }
        
    def select_random_tracks(self, audio_emb, current_stage, num_tracks):
        assert isinstance(audio_emb, dict),  'audio_emb must be dict'
        keys = list(audio_emb.keys())
        
        if current_stage > num_tracks:
            raise ValueError('current_stage cannot be greater than num_tracks')
        
        selected_keys = random.sample(keys, current_stage)
        selected_tensors = [audio_emb[key] for key in selected_keys]
        selected_audio_emb = torch.cat(selected_tensors, dim=1) if selected_tensors else None
        
        remaining_keys = [key for key in keys if key not in selected_keys]
        remaining_tensors = [audio_emb[key] for key in remaining_keys]
        remaining_audio_emb = torch.cat(remaining_tensors, dim=1) if remaining_tensors else None
        
        return selected_audio_emb, remaining_audio_emb, selected_keys, remaining_keys
    
    def create_prefix_prompt(self, selected_keys):
        token_mapping = {
            'bass': 'bass',
            'drums': 'drum',
            'other': 'other accompaniment',
        }
        
        task_tokens = [token_mapping.get(key, '') for key in selected_keys]
        combined_task_token = '[{}] generation'.format(' & '.join(task_tokens))
        
        return combined_task_token