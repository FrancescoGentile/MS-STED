##
##
##

from typing import Iterator

from torch.optim import Optimizer
from torch.nn.parameter import Parameter

from .. import utils

class OptimizerConfig: 
    def __init__(self, options: dict) -> None:
        self._name = options.name
        if self._name is None:
            raise ValueError(f'Missing name for optimizer.')
        
        if not utils.check_class_exists('torch.optim', self._name):
            raise ValueError(f'No optimizer with the given name was found.')
        
        del options.name
        self._args = options
    
    def to_optimizer(self, params: Iterator[Parameter]) -> Optimizer:
        cls  = utils.get_class_by_name('torch.optim', self._name)
        return cls(params=params, **self._args)