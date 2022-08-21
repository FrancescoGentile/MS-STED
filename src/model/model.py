##
##
##

from typing import List, Tuple, Union
import torch 
import torch.nn as nn

from .config import DecoderConfig, DropoutConfig, EncoderConfig
from .embeddings import Embeddings
from .modules import Block
from ..dataset.skeleton import SkeletonGraph

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig, 
                 skeleton: SkeletonGraph, 
                 save_intermediates: bool = False) -> None:
        super().__init__()
        
        self._save_interm = save_intermediates
        
        self.blocks = nn.ModuleList()
        for block_cfg in config.blocks:
            self.blocks.append(Block(block_cfg, skeleton))
        
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
            x = block(x)
            
            if self.save_intermediates:
                output.append(x)
            else:
                output = x

        return output
    
class Classifier(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 dropout: DropoutConfig) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout.feature)
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, M, C, T, V = x.shape 
        x = x.view(N, M, C, T * V)
        # (N, C)
        avg = x.mean(-1).mean(1)
        avg = self.dropout(avg)
        out = self.fc(avg)
        
        return out

class ClassificationModel(nn.Module):
    def __init__(self, embeddings: Embeddings, encoder: Encoder, classifier: Classifier) -> None:
        super().__init__()
        
        self.embeddings = embeddings
        self.encoder = encoder
        self.classifier = classifier
        
        self.encoder.save_intermediates = False
    
    def forward(self, joints: torch.Tensor, bones: torch.Tensor) -> torch.Tensor:
        # Change shape: (N, C, T, V, M) -> (N * M, C, T, V)
        N, C, T, J, M = joints.shape
        _, _, _, B, _ = bones.shape
        j = joints.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, J)
        b = bones.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, B)
        
        # Apply embeddings
        x: torch.Tensor = self.embeddings(j, b)
        # Apply encoder
        eout: torch.Tensor = self.encoder(x)
        
        # Apply
        _, C, T, V = eout.shape
        # (N, M, C_out, T, V)
        output = eout.view(N, -1, C, T, V)
        output = self.classifier(output)
        
        return output


class Decoder(nn.Module):
    def __init__(self, dconfig: DecoderConfig, econfig: EncoderConfig) -> None:
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.transforms = nn.ModuleList()
        for eblock, dblock in zip(econfig.blocks[::-1], dconfig.blocks):
            self.blocks.append(Block(dblock))
            if eblock.out_channels != dblock.in_channels:
                self.transforms.append(
                    nn.Conv2d(eblock.out_channels, dblock.in_channels, kernel_size=1))
            else:
                self.transforms.append(nn.Identity())
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        
        out = 0
        for idx, (block, tr, ex) in enumerate(zip(self.blocks, self.transforms, x[::-1])):
            ex = tr(ex)
            out = block(ex + out)
            
            if idx < len(self.blocks) - 1:
                # (N, C, T * 2, V)
                out = torch.repeat_interleave(out, 2, dim=2)

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels: int, dropout: DropoutConfig) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout.feature)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        out = self.conv(x)
        
        return out

class Reconstructor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: DropoutConfig) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout.feature)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        out = self.conv(x)
        
        return out

class ReconstructorDiscriminatorModel(nn.Module):
    def __init__(self, 
                 embeddings: Embeddings, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 reconstructor: Reconstructor, 
                 discriminator: Discriminator) -> None:
        super().__init__()
        
        self.embeddings = embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.reconstructor = reconstructor
        
        self.encoder.save_intermediates = True
        
    def forward(self, joints: torch.Tensor, bones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Change shape: (N, C, T, V, M) -> (N * M, C, T, V)
        N, C, T, J, M = joints.shape
        _, _, _, B, _ = bones.shape
        j = joints.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, J)
        b = bones.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, B)
        
        # Apply embeddings
        x: torch.Tensor = self.embeddings(j, b)
        # Apply encoder
        eout: List[torch.Tensor] = self.encoder(x)
        # Apply decoder
        dout: torch.Tensor = self.decoder(eout)
        
        recon_out: torch.Tensor = self.reconstructor(dout)
        disc_out: torch.Tensor = self.discriminator(dout)
        
        # Apply
        _, C, T, U = recon_out.shape
        recon_out = recon_out.view(N, -1, C, T, U)
        recon_out = recon_out.permute(0, 2, 3, 4, 1).contiguous()
        
        _, _, T, U = disc_out.shape
        disc_out = disc_out.view(N, -1, T, U)
        disc_out = disc_out.permute(0, 2, 3, 1).contiguous()
        
        jr, br = recon_out[:, :, :, :J], recon_out[:, :, :, J:]
        jd, bd = disc_out[:, :, :J], disc_out[:, :, J:]
        
        return jr, jd, br, bd