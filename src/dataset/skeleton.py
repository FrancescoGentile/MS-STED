##
##
##

import numpy as np

class SkeletonGraph:
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def joints_connections(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def joints_bones_adjacency_matrix(self) -> np.ndarray:
        raise NotImplementedError
    

def normalize_adjacency_matrix(adj: np.ndarray) ->np.ndarray:
    node_degrees = adj.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.identity(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ adj @ norm_degs_matrix).astype(np.float32)

def laplacian_adjacency_matrix(adj: np.ndarray) -> np.ndarray:
    identity = np.identity(len(adj))
    norm_adj = normalize_adjacency_matrix(adj)
    laplacian: np.ndarray = identity - norm_adj
        
    return laplacian