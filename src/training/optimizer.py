##
##
##


from torch.optim import Optimizer
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .. import utils

class OptimizerConfig: 
    def __init__(self, cfg: dict) -> None:
        self._name = cfg.name
        if self._name is None:
            raise ValueError(f'Missing name for optimizer.')
        
        if not utils.check_class_exists('torch.optim', self._name):
            raise ValueError(f'No optimizer with the given name was found.')
        
        self._args = cfg.args.toDict() if cfg.args is not None else {}
    
    def to_optimizer(self, model: nn.Module) -> Optimizer:
        cls  = utils.get_class_by_name('torch.optim', self._name)
        
        if isinstance(model, DDP):
            model = model.children().__next__()
        
        submodules = dict(model.named_children())
        params = []
        
        defaults = {}
        for key, value in self._args.items():
            if key in submodules:
                params.append({'params': submodules[key].parameters(), **value})
                del submodules[key]
            else:
                defaults[key] = value
        
        remaining = []
        for module in submodules.values():
            remaining += list(module.parameters())
           
        params.append({'params': remaining})
        
        return cls(params, **defaults)
                

    def to_dict(self) -> dict:
        d = {'name': self._name,
             'args': self._args}
        
        return d
    
    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def args(self) -> dict:
        return self._args