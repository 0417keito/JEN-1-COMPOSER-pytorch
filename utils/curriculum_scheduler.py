class CurriculumScheduler:
    def __init__(self, total_epochs, stage_ratios):
        self.total_epochs = total_epochs
        self.stage_ratios = stage_ratios
        self.curriculum_stages = len(stage_ratios)
        self.stage_epochs = self.calculate_stage_epochs()
        self.current_stage = 1
    
    def calculate_stage_epochs(self):
        stage_epochs = []
        for ratio in self.stage_ratios:
            epochs = round(self.total_epochs * ratio)
            stage_epochs.append(epochs)
        return stage_epochs
    
    def update_stage(self):
        if self.current_stage < self.curriculum_stages:
            self.current_stage += 1
    
    def get_current_stage_epochs(self):
        start_epoch = sum(self.stage_epochs[:self.current_stage - 1])
        end_epoch = start_epoch + self.stage_epochs[self.current_stage -1]
        return start_epoch, end_epoch
    
if __name__ == '__main__':
    total_epochs = 100
    stage_ratios = [0.1, 0.2, 0.3, 0.2, 0.2]
    
    curriculum_scheduler = CurriculumScheduler(total_epochs, stage_ratios)