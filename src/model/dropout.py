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
        self._middle_step = math.floor(num_steps / 2)
        self._delta = p / self._middle_step
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
        elif self._step > self._middle_step:
            for m in self._modules:
                m.structure_dropout += self._delta
        else:
            for m in self._modules:
                m.structure_dropout -= self._delta