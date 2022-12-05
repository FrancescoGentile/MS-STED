##
##
##

import math
from typing import List

import torch.nn as nn

from .modules import SpatioTemporalAttention

class DropHeadScheduler:
    def __init__(self, 
                 modules: List[nn.Module], 
                 start: float,
                 end: float,
                 warmup: int,
                 num_epochs: int, 
                 steps_per_epoch: int,
                 start_epoch: int = 0) -> None:
        
        self._num_steps = num_epochs * steps_per_epoch
        self._change_step = self._num_steps // warmup if warmup != 0 else 0
        self._start = start
        self._delta1 = start / self._change_step if self._change_step > 0 else 0
        self._delta2 = end / (self._num_steps - self._change_step)
        self._step = start_epoch * steps_per_epoch
        
        current_p = self._compute_dropout()
        self._modules: List[SpatioTemporalAttention] = []
        for m in modules:
            if isinstance(m, SpatioTemporalAttention):
                self._modules.append(m)
                m.structure_dropout = current_p
    
    def _compute_dropout(self) -> float:
        if self._step >= self._change_step:
            num_steps = self._step - self._change_step
            new_p = self._delta2 * num_steps
        else:
            new_p = self._start - self._delta1 * self._step
        
        return new_p
    
    def step(self):
        self._step += 1
        current_p = self._compute_dropout()
        for m in self._modules:
            m.structure_dropout = current_p