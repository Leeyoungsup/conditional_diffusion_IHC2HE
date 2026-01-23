from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    # Initialize the scheduler
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler = None, last_epoch = None):
        self.multiplier = multiplier  # Maximum learning rate multiplier
        self.total_epoch = warm_epoch  # Number of epochs for the warm-up
        self.after_scheduler = after_scheduler  # Scheduler to use after the warm-up
        self.finished = False  # Flag to indicate if the warm-up has finished
        self.last_epoch = last_epoch  # The last epoch number
        self.base_lrs = None  # Base learning rates
        super().__init__(optimizer)

    # Compute the learning rate for the current epoch
    def get_lr(self):
        # If the warm-up is finished
        if self.last_epoch > self.total_epoch:
            # If there is an after scheduler
            if self.after_scheduler:
                # If the warm-up has not been finished
                if not self.finished:
                    # Update the base learning rates of the after scheduler
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True  # Set the finished flag to True
                # Return the learning rates from the after scheduler
                return self.after_scheduler.get_last_lr()
            # If no after scheduler, return the scaled base learning rates
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        # If in the warm-up period, calculate the learning rate based on the current epoch
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    # Save the state of the scheduler
    def state_dict(self):
        # Get the state dict of the warm-up scheduler
        warmdict = {key:value for key, value in self.__dict__.items() if (key != 'optimizer' and key != 'after_scheduler')}
        # Get the state dict of the after scheduler
        cosdict = {key:value for key, value in self.after_scheduler.__dict__.items() if key != 'optimizer'}
        # Return both dictionaries as a single state dict
        return {'warmup':warmdict, 'afterscheduler':cosdict}

    # Load the state of the scheduler
    def load_state_dict(self, state_dict: dict):
        # Update the after scheduler and warm-up scheduler states
        self.after_scheduler.__dict__.update(state_dict['afterscheduler'])
        self.__dict__.update(state_dict['warmup'])

    # Step the scheduler
    def step(self, epoch=None, metrics=None):
        # If the warm-up is finished and there is an after scheduler
        if self.finished and self.after_scheduler:
            # If no specific epoch is given, step the after scheduler with None
            if epoch is None:
                self.after_scheduler.step(None)
            # Otherwise, step the after scheduler with the adjusted epoch
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        # If still in the warm-up or no after scheduler, step the warm-up scheduler
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
