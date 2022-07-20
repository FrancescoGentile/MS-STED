##
##
##

import torch.utils.data as data

class Dataset(data.Dataset):
    
    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        raise NotImplementedError
    
    @property
    def channels(self) -> int:
        raise NotImplementedError
    
    @property
    def num_frames(self) -> int:
        raise NotImplementedError