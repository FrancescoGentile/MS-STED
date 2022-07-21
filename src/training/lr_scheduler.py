##
##
##

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import timm.scheduler

from .. import utils

class LRSchedulerConfig:
    def __init__(self, options: dict) -> None:
        self._name = options.name
        
        if self._name is None:
            raise ValueError(f'Missing name for lr scheduler.')
        
        if self._name != 'CosineLRScheduler' and not utils.check_class_exists('torch.optim.lr_scheduler', self._name):
            raise ValueError(f'No lr scheduler with the given name was found.')
        
        del options.name 
        
        if options.after_batch is not None:
            self._after_batch = options.after_batch
            del options.after_batch
        else: 
            self._after_batch = False
            
        self._args = options
        
    def to_lr_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        if self._name == 'CosineLRScheduler':
            cls = timm.scheduler.CosineLRScheduler
        else:
            cls =  utils.get_class_by_name('torch.optim.lr_scheduler', self._name)
        return cls(optimizer=optimizer, **self._args)
    
    @property
    def after_batch(self) -> bool:
        return self._after_batch