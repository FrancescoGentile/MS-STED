##
##
##

import numpy as np
from ..skeleton import SkeletonGraph

class NTUSkeletonGraph(SkeletonGraph):
    
    def __init__(self) -> None:
        super().__init__()
        
        self._num_joints = 25
        
        links = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                 (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                 (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                 (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                 (22, 23), (23, 8), (24, 25), (25, 12)]

        self_links = [(i, i) for i in range(self._num_joints)]
        edges = [(i - 1, j - 1) for (i, j) in links]
        self._edges = edges + self_links
        
        connections = np.array([(1, 2), (2, 2), (3, 21), (4, 3), (5, 21), 
                                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), 
                                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), 
                                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), 
                                (21, 2), (22, 23), (23, 8), (24, 25), (25, 12)])
        self._connections = connections - (1, 1)
        
        self._adjacency = self._get_adjacency_matrix()
        self._joints_bones = self._get_joints_and_bones()
    
    def _get_adjacency_matrix(self) -> np.ndarray:
        adj = np.zeros((self._num_joints, self._num_joints))
        for i, j in self._edges:
            adj[j, i] = 1
            adj[i, j] = 1
        
        return adj
    
    def _get_joints_and_bones(self) -> np.ndarray:
        num_bones = len(self._connections)
        adj = np.zeros((self._num_joints + num_bones, self._num_joints + num_bones))
        for i,j in self._edges:
            adj[i, j] = 1
            adj[j, i] = 1
        
        for idx, bone in enumerate(self._connections):
            bone_idx = idx + self._num_joints
            adj[bone_idx, bone_idx] = 1
            adj[bone_idx, bone[0]] = 1
            adj[bone[0], bone_idx] = 1
            adj[bone_idx, bone[1]] = 1
            adj[bone[1], bone_idx] = 1
        
        return adj
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        return self._adjacency
    
    @property
    def joints_connections(self) -> np.ndarray:
        return self._connections
    
    @property
    def joints_bones_adjacency_matrix(self) -> np.ndarray:
        return self._joints_bones
    