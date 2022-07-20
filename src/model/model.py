##
##
##

from typing import List, Union
import torch 
import torch.nn as nn

from .config import EncoderConfig
from .embeddings import Embeddings
from .modules import Block
from .tools import init_layers

class Encoder(nn.Module):
    
    def __init__(self, config: EncoderConfig, save_intermediates: bool = False) -> None:
        super().__init__()
        
        self._save_interm = save_intermediates
        
        self.blocks = nn.ModuleList()
        for block_cfg in config.blocks:
            self.blocks.append(Block(block_cfg))
        
        self.pooling = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        #self.apply(init_layers)
        
    @property
    def save_intermediates(self) -> bool:
        return self._save_interm
    
    @save_intermediates.setter
    def save_intermediates(self, new_save: bool):
        self._save_interm = new_save
    
    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        
        if self.save_intermediates:
            output = []
        else:
            output = None
        
        # Apply blocks
        for block in self.blocks:
            tmp = block(x)
            x = self.pooling(tmp)
            
            if self.save_intermediates:
                output.append(tmp)
            else:
                output = tmp

        return output
    
class Classifier(nn.Module):
    
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)
        
        self.apply(init_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, M, C, T, V = x.shape 
        x = x.view(N, M, C, T * V)
        # (N, C)
        avg = x.mean(-1).mean(1)
        avg = self.dropout(avg)
        out = self.fc(avg)
        
        return out

class EncoderClassifier(nn.Module):
    def __init__(self, embeddings: Embeddings, encoder: Encoder, classifier: Classifier) -> None:
        super().__init__()
        
        self.embeddings = embeddings
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, joints: torch.Tensor, bones: torch.Tensor) -> torch.Tensor:
        # Change shape: (N, C, T, V, M) -> (N * M, C, T, V)
        N, C, T, V, M = joints.shape
        j = joints.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        b = bones.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Apply embeddings
        x: torch.Tensor = self.embeddings(j, b)
        # Apply encoder
        x: torch.Tensor = self.encoder(x)
        
        # Apply
        _, C, T, V = x.shape
        # (N, M, C_out, T, V)
        output = x.view(N, -1, C, T, V)
        output = self.classifier(output)
        
        return output