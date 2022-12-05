##
##
##

import numpy as np

class SkeletonGraph:
    
    def adjacency_matrix(self, self_loops: bool) -> np.ndarray:
        raise NotImplementedError
    
    def bones(self) -> np.ndarray:
        raise NotImplementedError