##
##
##

import math
from typing import List

import torch.nn as nn

from .modules import SpatioTemporalAttention

class StructureDropoutScheduler:
    def __init__(self, modules: List[nn.Module], p: float, num_steps: int) -> None:
        self._num_steps = num_steps
        self._change_step = math.floor(num_steps / 8)
        self._delta1 = p / self._change_step
        self._delta2 = p / (self._num_steps - self._change_step)
        self._step = 0
        
        self._modules: List[SpatioTemporalAttention] = []
        for m in modules:
            if isinstance(m, SpatioTemporalAttention):
                self._modules.append(m)
                m.structure_dropout = p
    
    def step(self):
        self._step += 1
        if self._step > self._num_steps:
            pass
        elif self._step > self._change_step:
            for m in self._modules:
                m.structure_dropout += self._delta2
        else:
            for m in self._modules:
                m.structure_dropout -= self._delta1