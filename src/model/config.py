##
##
##

from __future__ import annotations
from typing import List

from .. import utils

# Configuration for model
class ModelConfig:
    def __init__(self, cfg: dict) -> None:
        self._name = cfg.name
        
        if cfg.architecture is None:
            raise ValueError('Missing architecture field in model config')
        model_cfg = utils.load_config_file(cfg.architecture)
        
        if model_cfg.embeddings is None:
            raise ValueError('Missing embeddings field in model config')
        self._embeddings = EmbeddingsConfig(model_cfg.embeddings)
        
        if model_cfg.encoder is None:
            raise ValueError('Missing encoder field in model config')
        self._encoder = EncoderConfig(model_cfg.encoder, self._embeddings.out_channels)
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def embeddings(self) -> EmbeddingsConfig:
        return self._embeddings
    
    @property
    def encoder(self) -> EncoderConfig:
        return self._encoder

# Configuration for embeddings
class EmbeddingsConfig:
    def __init__(self, cfg: dict) -> None:
        self._out_channels = cfg.channels
        if cfg.channels is None:
            raise ValueError('Missing channels field in embeddings config')
        elif type(cfg.channels) != int:
            raise ValueError('Channels field in embeddings config must be an integer')
        
        self._temporal_channels = cfg.temporal_channels
        if cfg.temporal_channels is None:
            raise ValueError('Missing temporal-channels field in embeddings config')
        elif type(cfg.temporal_channels) != int:
            raise ValueError('Temporal-channels field in embeddings config must be an integer')
        
        self._type_channels = cfg.type_channels
        if cfg.type_channels is None:
            raise ValueError('Missing type-channels field in embeddings config')
        elif type(cfg.type_channels) != int:
            raise ValueError('Type-channels field in embeddings config must be an integer')
    
    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    @property
    def temporal_channels(self) -> int:
        return self._temporal_channels
    
    @property
    def type_channels(self) -> int:
        return self._type_channels

# Configuration for encoder
class EncoderConfig:
    def __init__(self, cfg: dict, in_channels: int) -> None:    
        self._in_channels = in_channels
        
        if cfg.blocks is None:
            raise ValueError('Missing blocks config')
        elif type(cfg.blocks) != list:
            raise ValueError('Blocks config must be a list')
        
        self._blocks: List[BlockConfig] = []
        in_ch = in_channels
        for bc in cfg.blocks:
            block = BlockConfig(bc, in_channels=in_ch)
            in_ch = block.out_channels
            self._blocks.append(block)
    
    @property
    def in_channels(self) -> int:
        return self._in_channels
    
    @property
    def out_channels(self) -> int:
        return self._blocks[-1].out_channels
    
    @property
    def blocks(self) -> List[BlockConfig]:
        return self._blocks

# Configuration for a block
class BlockConfig:
    def __init__(self, cfg: dict, in_channels: int) -> None:
        # Set channels
        if cfg.channels is None:
            raise ValueError('Missing channels field in block config')
        elif type(cfg.channels) != int:
            raise ValueError('Channel field in block config must be an integer')
        
        self._out_channels = cfg.channels
        self._in_channels = in_channels
        
        # Set layers
        if cfg.layers is None:
            raise ValueError('Missing layers field in block config')
        elif type(cfg.layers) != list:
            raise ValueError('Layers field in block confif must be a list')
        
        self._layers: List[LayerConfig] = []
        in_ch = in_channels
        out_ch = self.out_channels
        for lc in cfg.layers:
            self._layers.append(LayerConfig(lc, in_ch, out_ch))
            in_ch = out_ch
    
    @property
    def in_channels(self) -> int:
        return self._in_channels
    
    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    @property
    def layers(self) -> List[LayerConfig]:
        return self._layers


# Configuration for a layer
class LayerConfig:
    def __init__(self, cfg: dict, in_channels: int, out_channels: int) -> None:
        # Set channels
        self._in_channels = in_channels
        self._out_channels = out_channels
        
        # Set cross view
        if cfg.cross_view is None:
            raise ValueError('Missing cross-view field in layer config')
        elif type(cfg.cross_view) != bool:
            raise ValueError('Cross-view field in layer config must be a boolean')
        
        self._cross_view = cfg.cross_view
        
        # Set branches
        if cfg.branches is None:
            raise ValueError('Missing branches field in layer config')
        elif type(cfg.branches) != list:
            raise ValueError('Branches field in layer confif must be a list')
        
        self._branches: List[LayerConfig] = []
        for bc in cfg.branches:
            self._branches.append(BranchConfig(bc, in_channels, out_channels))
            
    @property
    def in_channels(self) -> int:
        return self._in_channels
    
    @property
    def out_channels(self) -> int:
        return self._out_channels
        
    @property
    def cross_view(self) -> bool:
        return self._cross_view
    
    @property
    def branches(self) -> List[BranchConfig]:
        return self._branches

class BranchConfig:
    def __init__(self, cfg: dict, in_channels: int, out_channels: int) -> None:
        # Set channels
        self._in_channels = in_channels
        self._out_channels = out_channels
        
        # Set number of heads
        self._num_heads = cfg.num_heads
        if cfg.num_heads is None:
            raise ValueError('Missing num-heads field in branch config')
        elif type(cfg.num_heads) != int:
            raise ValueError('Num-heads field in branch config must be an integer')
        elif out_channels % cfg.num_heads != 0:
            raise ValueError('Number of heads in a branch must divide the number of output channels')
        
        # Set window
        if cfg.window is None:
            raise ValueError('Missing window field in branch config')
        elif type(cfg.window) != int:
            raise ValueError('Window field in branch config must be an integer')
        
        self._window = cfg.window
        
        # Set dilation
        if cfg.dilation is None:
            raise ValueError('Missing dilation field in branch config')
        elif type(cfg.dilation) != int:
            raise ValueError('Dilation field in branch config must be an integer')
        
        self._dilation = cfg.dilation
        
    @property
    def in_channels(self) -> int:
        return self._in_channels
    
    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    @property
    def num_heads(self) -> int:
        return self._num_heads
    
    @property
    def window(self) -> int:
        return self._window
    
    @property
    def dilation(self) -> int:
        return self._dilation
    
