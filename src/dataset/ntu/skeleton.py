##
##
##

from typing import Final

import numpy as np
from ..skeleton import SkeletonGraph

class NTUSkeletonGraph(SkeletonGraph):
    _num_joints: Final[int] = 25
    _bones: Final[np.ndarray]
    
    def __init__(self) -> None:
        super().__init__()
        
        links = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                 (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                 (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                 (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                 (8, 22), (7, 23), (11, 25), (12, 24)]
        
        self._bones = np.array(links) - np.array([1, 1])
    
    def adjacency_matrix(self, self_loops: bool) -> np.ndarray:
        adj = np.zeros((self._num_joints, self._num_joints))
        for (i, j) in self._bones:
            adj[i, j] = adj[j, i] = 1
            
        if self_loops:
            adj = adj + np.identity(self._num_joints)
        
        return adj
    
    def bones(self) -> np.ndarray:
        return self._bones
    