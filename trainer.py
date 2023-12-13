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
                 cross_attn_cond_ids=['prompt'],
                 global_cond_ids= [],
                 input_concat_ids= ['masked_input', 'mask'],
                 composer=False,
                 curriculum_scheduler: CurriculumScheduler = None
                 ):
        super().__init__()
        self.config=config
        self.tasks = self.config.tasks
        self.rank = rank
        self.epoch_str = epoch_str
        self.global_step = global_step
        self.grad_clip = grad_clip
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
        
        self.composer = composer
        self.curriculum_scheduler = curriculum_scheduler
        
    def curriculum_train(self):
        if self.composer:
            assert self.curriculum_scheduler is not None, "Curriculum scheduler is not provided"
            num_curriculum_stages = self.curriculum_scheduler.curriculum_stages
            self.curriculum_scheduler.find_stage_for_epoch(self.epoch_str)
            
            for stage in range(self.curriculum_scheduler.current_stage, num_curriculum_stages + 1):
                start_epoch, end_epoch = self.curriculum_scheduler.get_current_stage_epochs(stage)
                self.train_loop(max(self.epoch_str, start_epoch), end_epoch)
                self.curriculum_scheduler.update_stage()
        else:
            self.train_loop(self.epoch_str, self.config.num_epoch)
        
    def eval_all_tasks(self, rank=0):
        task_losses = {task: 0.0 for task in self.tasks}
        num_batches = {task: 0 for task in self.tasks}
        total_loss = 0.0
        total_batches = 0
        
        for task in self.tasks:
            task_loss, task_count = self.eval(task=task, rank=rank)
            task_losses[task] = task_loss
            num_batches[task] = task_count
            total_loss += task_loss
            total_batches += task_count
        
        for task, loss_sum in task_losses.items():
            avg_loss = loss_sum / num_batches[task] if num_batches[task] > 0 else 0
            self.logger.info(f'Average validation loss for task {task}: {avg_loss}')
            if self.rank == 0:
                scalars = {f'loss/val_{task}': avg_loss}
                summarize(writer=self.writer, global_step=self.global_step, scalars=scalars)
        
        avg_total_loss = total_loss / total_batches if total_batches > 0 else 0
        self.logger.info(f'Average total validation loss: {avg_total_loss}')
        if self.rank == 0:
            scalars = {'loss/val_total': avg_total_loss}
            summarize(writer=self.writer, global_step=self.global_step, scalars=scalars)
        
        self.model.train()
    
    def eval(self, task, rank=0):
        self.model.eval()
        count = 0
        loss_sum = 0
        with torch.no_grad():
            for batch_idx, (audio_emb, metadata, demix_embs_dict) in enumerate(self.valid_dl):
                b, _, _, device = *audio_emb.shape, self.config.device
                masked_input, mask, causal = self.random_mask(audio_emb, audio_emb.shape[2], task)
                conditioning = self.conditioner(metadata, self.config.device)
                conditioning['masked_input'] = masked_input
                conditioning['mask'] = mask
                conditioning = self.get_conditioning(conditioning)
                if self.config.diffusion_type == 'gdm':
                    num_timesteps = self.diffusion.num_timesteps
                    t = torch.randint(0, num_timesteps, (b,), device=device).long()
                    with autocast(enabled=self.config.use_fp16):
                        loss = self.diffusion.training_loosses(self.model, audio_emb, t, conditioning, causal=causal)
                else:
                    with autocast(enabled=self.config.use_fp16):
                        loss = self.diffusion.training_loosses(self.model, audio_emb, conditioning, causal=causal)
                    
                loss_sum += loss
                count += 1
                
        return loss_sum, count
        
    def train_loop(self, str_epoch, end_epoch):
        for epoch in range(str_epoch, int(end_epoch + 1)):
            total_batches = len(self.train_dl)
            batches_per_task = total_batches // len(self.tasks)
            assert batches_per_task >= 1, 'batches_per_task must be greater than or equal to 1'
            data_iter = iter(self.train_dl)
            
            for i in range(batches_per_task):
                weighted_loss = 0.0
                loss_dict = {}
                for batch_idx, (audio_emb, metadata, demix_embs_dict) in enumerate(data_iter):
                    shape = audio_emb.shape
                    if self.composer and demix_embs_dict is not None:
                        num_tracks = len(demix_embs_dict)
                        audio_emb = demix_embs_dict
                    else:
                        num_tracks = 1
                    if batch_idx >= batches_per_task:
                        break
                    weighted_loss = 0.0
                    loss_dict = {}
                    for task in self.tasks:
                        loss = self.train(task=task, audio_emb=audio_emb, metadata=metadata, num_tracks=num_tracks, shape=shape)
                        weighted_loss += loss
                        loss_dict[task] = loss.item()
                
                self.optimizer.zero_grad()
                self.scaler.scale(weighted_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.lr_scheduler.step()
                self.scaler.update()
                
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
                            f'loss: {weighted_loss} '
                            f'loss_text_guided: {loss_text_guided} '
                            f'loss_inpaint: {loss_inpaint} '
                            f'loss_cont: {loss_cont} '
                            f'global_step: {self.global_step}, lr:{lr}')
                        scalars = {'loss/train': weighted_loss,
                                   'loss_text_guided/train': loss_text_guided,
                                   'loss_inpaint/train': loss_inpaint,
                                   'loss_cont/train': loss_cont}
                        summarize(writer=self.writer, global_step=self.global_step, scalars=scalars)
                        
                    if self.global_step % self.config.eval_interval == 0:
                        self.eval_all_tasks(rank=self.rank)
                        save_checkpoint(model=self.model, optimizer=self.optimizer,
                                        lr=self.config.optimizer_config.lr, iteration=epoch,
                                        checkpoint_path=os.path.join(self.config.save_dir, f'Jen1_step_{self.global_step}.pth'),
                                        logger=self.logger)
                     
                self.global_step += 1   
    
    def train(self, task, audio_emb, metadata, num_tracks=1, shape=None):
        self.model.train()
        b, c, _, device = *shape, self.config.device
        
        if self.composer and num_tracks != 1:
            current_stage = self.curriculum_scheduler.current_stage 
            selected_audio_emb, remaining_audio_emb, selected_keys, remaining_keys = \
                self.select_random_tracks(audio_emb, current_stage, num_tracks)
            prefix_prompt = self.create_prefix_prompt(selected_keys)
            for item in metadata:
                item['prompt'] = prefix_prompt + ' ' + item['prompt']
        
        masked_input, mask, causal = self.random_mask(audio_emb, audio_emb.shape[2], task)
        conditioning = self.conditioner(metadata, self.config.device)
        conditioning['masked_input'] = masked_input
        conditioning['mask'] = mask
        conditioning = self.get_conditioning(conditioning)
        
        if self.config.diffusion_type == 'gdm':
            num_timesteps = self.diffusion.num_timesteps
            if self.composer:
                t_i = torch.randint(1, num_timesteps-1, (b,), device=device).long()
                t_for_cond = torch.zeros(b, dtype=torch.long, device=device)
                for i in range(b):
                    t_for_cond[i] = random.choice([0, t_i[i], num_timesteps])
                with autocast(enabled=self.config.use_fp16):
                    loss = self.diffusion.training(self.model, (selected_audio_emb, remaining_audio_emb), 
                                                   (t_i, t_for_cond), conditioning, causal=causal)
            else:
                t = torch.randint(0, num_timesteps, (b,), device=device).long()
                with autocast(enabled=self.config.use_fp16):
                    loss = self.diffusion.training_loosses(self.model, audio_emb, t, conditioning, causal=causal)
        else:
            times = torch.rand((b,)).requires_grad_(True).to(device=device)
            with autocast(enabled=self.config.use_fp16):
                loss = self.diffusion.training_loosses(self.model, audio_emb, times, conditioning, causal=causal)
        return loss
            
    def random_mask(self, sequence, max_mask_length, task):
        b, _, sequence_length = sequence_length.size()
        
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
        
    def select_random_tracks(audio_emb, current_stage, num_tracks):
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
            'vocals': 'vocal'
        }
        
        task_tokens = [token_mapping.get(key, '') for key in selected_keys]
        combined_task_token = '[{}] generation'.format(' & '.join(task_tokens))
        
        return combined_task_token