##
##
##

import numpy as np
from ..skeleton import SkeletonGraph

class NTUSkeletonGraph(SkeletonGraph):
    
    def __init__(self) -> None:
        super().__init__()
        
        self._num_joints = 25
        
        #links = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        #         (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        #         (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        #         (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        #         (22, 23), (23, 8), (24, 25), (25, 12)]
        #
        #self._edges = [(i - 1, j - 1) for (i, j) in links]
        #bones = np.array([
        #    (1, 2), (2, 2), (3, 21), (4, 3), (5, 21), 
        #    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), 
        #    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), 
        #    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), 
        #    (21, 2), (22, 23), (23, 8), (24, 25), (25, 12)])
        #self._bones = bones - (1, 1)
         
        bones = np.array([
            (2, 1), (1, 13), (1, 17), (13, 14), (14, 15),
            (15, 16), (17, 18), (18, 19), (19, 20), (2, 21),
            (21, 3), (3, 4), (21, 9), (9, 10), (10, 11),
            (11, 12), (12, 24), (12, 25), (21, 5), (5, 6),
            (6, 7), (7, 8), (8, 22), (8, 23)
        ])
        self._bones = bones - (1, 1)
    
    def joints_adjacency(self, self_edges: bool) -> np.ndarray:
        adj = np.zeros((self._num_joints, self._num_joints))
        for (i, j) in self._bones:
            adj[i, j] = adj[j, i] = 1
            
        if self_edges:
            adj = adj + np.identity(self._num_joints)
        
        return adj
    
    def joints_bones_adjacency(self, self_edges: bool) -> np.ndarray:
        J = self._num_joints
        B = len(self._bones)
        
        adj = np.zeros((J + B, J + B))
        adj[:J, :J] = self.joints_adjacency(False)
        
        for idx, (u, v) in enumerate(self._bones):
            adj[u, J + idx] = 1
            adj[J + idx, u] = 1
            adj[v, J + idx] = 1
            adj[J + idx, v] = 1
        
        if self_edges:
            adj = adj+ np.identity(J + B)
        
        return adj
    
    def bones(self) -> np.ndarray:
        return self._bones
    