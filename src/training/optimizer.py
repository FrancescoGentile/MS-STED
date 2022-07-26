##
##
##

from typing import Iterator

from torch.optim import Optimizer
from torch.nn.parameter import Parameter

from .. import utils

class OptimizerConfig: 
    def __init__(self, cfg: dict) -> None:
        self._name = cfg.name
        if self._name is None:
            raise ValueError(f'Missing name for optimizer.')
        
        if not utils.check_class_exists('torch.optim', self._name):
            raise ValueError(f'No optimizer with the given name was found.')
        
        self._args = cfg.args.toDict() if cfg.args is not None else {}
    
    def to_optimizer(self, params: Iterator[Parameter]) -> Optimizer:
        cls  = utils.get_class_by_name('torch.optim', self._name)
        return cls(params=params, **self._args)

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