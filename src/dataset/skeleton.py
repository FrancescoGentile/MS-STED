##
##
##

import numpy as np

class SkeletonGraph:
    
    def joints_adjacency(self, self_edges: bool) -> np.ndarray:
        raise NotImplementedError
    
    def joints_bones_adjacency(self, self_edges: bool) -> np.ndarray:
        raise NotImplementedError
    
    def bones(self) -> np.ndarray:
        raise NotImplementedError


def symmetric_normalized_adjacency(adj: np.ndarray) ->np.ndarray:
    node_degrees = adj.sum(-1)
    degs_inv_sqrt = 1 / np.sqrt(node_degrees)
    norm_degs_matrix = np.diag(degs_inv_sqrt)
    return (norm_degs_matrix @ adj @ norm_degs_matrix).astype(np.float32)