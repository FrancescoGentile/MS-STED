##
##
##

import numpy as np

class SkeletonGraph:
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def laplacian_matrix(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def joints_connections(self) -> np.ndarray:
        raise NotImplementedError