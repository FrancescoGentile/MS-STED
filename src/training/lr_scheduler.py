##
##
##

from __future__ import annotations
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

from .. import utils

class LRSchedulerConfig:
    def __init__(self, cfg: dict) -> None:
        self._name = cfg.name
        
        if self._name is None:
            raise ValueError(f'Missing name for lr scheduler.')
        elif self._name != 'CosineLRScheduler' and not utils.check_class_exists('torch.optim.lr_scheduler', self._name):
            raise ValueError(f'No lr scheduler with the given name was found.')
        
        self._step_after_batch = False
        if cfg.step_after_batch is not None:
            self._step_after_batch = cfg.step_after_batch
            
        self._args: dict = cfg.args.toDict() if cfg.args is not None else {}
        
    def to_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        if self._name == 'CosineLRScheduler':
            return TimmLRScheduler(self, optimizer)
        else:
            return PyTorchLRScheduler(self, optimizer)
    
    def to_dict(self) -> dict:
        d = {'name': self._name, 
             'step_after_batch': self._step_after_batch, 
             'args': self._args}
        
        return d
    
    def __str__(self) -> str:
        return self._name
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def step_after_batch(self) -> bool:
        return self._step_after_batch
    
    @property
    def args(self) -> dict:
        return self._args
    
class LRScheduler:
    def __init__(self, config: LRSchedulerConfig, optimizer: Optimizer) -> None:
        self._config = config
    
    def step(self, after_batch: bool):
        """Updates the learning rate according to the schedule

        Args:
            after_batch (bool): if True this is called after a batch is processed, otherwise at the end of the epoch
        """
        raise NotImplementedError
    
    def get_lr(self) -> float:
        raise NotImplementedError
    
    def state_dict(self) -> dict:
        raise NotImplementedError
    
    def load_state_dict(self, state_dict: dict):
        raise NotImplementedError


class PyTorchLRScheduler(LRScheduler):
    def __init__(self, config: LRSchedulerConfig, optimizer: Optimizer) -> None:
        super().__init__(config, optimizer)
        
        cls =  utils.get_class_by_name('torch.optim.lr_scheduler', config._name)
        self._scheduler: _LRScheduler = cls(optimizer, **config.args)
    
    def get_lr(self) -> float:
        return self._scheduler.get_last_lr()[-1]
    
    def step(self, after_batch: bool):
        if self._config.step_after_batch and after_batch:
            self._scheduler.step()
        elif not self._config.step_after_batch and not after_batch:
            self._scheduler.step()
            
    def state_dict(self) -> dict:
        return self._scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict):
        self._scheduler.load_state_dict(state_dict)


class TimmLRScheduler(LRScheduler):
    def __init__(self, config: LRSchedulerConfig, optimizer: Optimizer) -> None:
        super().__init__(config, optimizer)
        self._epoch = 0
        self._scheduler = CosineLRScheduler(optimizer, **config.args)
    
    def get_lr(self) -> float:
        return self._scheduler.optimizer.param_groups[0]['lr']
    
    def step(self, after_batch: bool):
        if not after_batch:
            self._epoch += 1
            self._scheduler.step(self._epoch)
            
    def state_dict(self) -> dict:
        return self._scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict):
        self._scheduler.load_state_dict(state_dict)