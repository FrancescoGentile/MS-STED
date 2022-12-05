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
        
        self._sublayer_dropout = model_cfg.sublayer_dropout
        if self._sublayer_dropout is None:
            self._sublayer_dropout = 0.0
        
        self._feature_dropout = model_cfg.feature_dropout
        if self._feature_dropout is None:
            self._feature_dropout = 0.0
            
        self._structure_dropout = model_cfg.structure_dropout
        if self._structure_dropout is None:
            self._structure_dropout = 0.0
        
        if model_cfg.encoder is None:
            raise ValueError('Missing encoder field in model config')
        self._encoder = EncoderConfig(
            model_cfg.encoder, 
            self._feature_dropout,
            self._sublayer_dropout)
        
        if model_cfg.decoder is None:
            raise ValueError('Missing decoder field in model config')
        self._decoder = DecoderConfig(
            model_cfg.decoder,
            len(self.encoder.blocks),
            self._feature_dropout,
            self._sublayer_dropout)
        
        if model_cfg.embeddings is None:
            raise ValueError('Missing embeddings field in model config')
        self._embeddings = EmbeddingsConfig(
            model_cfg.embeddings, 
            out_channels=self.encoder.in_channels)
        
    def to_dict(self, architecture: bool) -> dict:
        """_summary_

        Args:
            architecture (bool): whether to add the architecture to the dictionary

        Returns:
            dict: _description_
        """
        d = {'name': self._name }
        
        if architecture:
            d['sublayer_dropout'] = self._sublayer_dropout
            d['feature_dropout'] = self._feature_dropout
            d['structure_dropout'] = self._structure_dropout
            d['embeddings'] = self._embeddings.to_dict()
            d['encoder'] = self._encoder.to_dict()
            d['decoder'] = self._decoder.to_dict()
        
        return d
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def embeddings(self) -> EmbeddingsConfig:
        return self._embeddings
    
    @property
    def encoder(self) -> EncoderConfig:
        return self._encoder
    
    @property
    def decoder(self) -> DecoderConfig:
        return self._decoder
    
    @property
    def feature_dropout(self) -> float:
        return self._feature_dropout
    
    @property
    def structure_dropout(self) -> float:
        return self._structure_dropout

# Configuration for embeddings
class EmbeddingsConfig:
    def __init__(self, cfg: dict, out_channels: int) -> None:
        self._out_channels = out_channels
        
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
        
        self._id_channels = cfg.id_channels
        if cfg.id_channels is None:
            raise ValueError('Missing id-channels field in embeddings config')
        elif type(cfg.id_channels) != int:
            raise ValueError('Id-channels field in embeddings config must be an integer')
    
    def to_dict(self) -> dict:
        d = {'temporal-channels': self._temporal_channels, 
             'type-channels': self._type_channels, 
             'out-channels': self._out_channels}
        
        return d
    
    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    @property
    def temporal_channels(self) -> int:
        return self._temporal_channels
    
    @property
    def type_channels(self) -> int:
        return self._type_channels
    
    @property
    def id_channels(self) -> int:
        return self._id_channels

# Configuration for encoder
class EncoderConfig:
    def __init__(self, cfg: dict, fdrop: float,sdrop: float) -> None:
        if cfg.blocks is None:
            raise ValueError('Missing blocks field in encoder config')
        elif type(cfg.blocks) != list:
            raise ValueError('Blocks field in encoder config must be a list')
        
        self._blocks: List[BlockConfig] = [None] * len(cfg.blocks)
        out_ch = -1
        for idx, bc in reversed(list(enumerate(cfg.blocks))):
            block = BlockConfig(bc, out_channels=out_ch, fdrop=fdrop, sdrop=sdrop)
            out_ch = block.in_channels
            self._blocks[idx] = block
    
    def to_dict(self) -> dict:
        d = {'in-channels': self.in_channels,
             'out-channels': self.out_channels}
        
        blocks = []
        for b in self._blocks:
            blocks.append(b.to_dict())
        
        d['blocks'] = blocks
        
        return d
    
    @property
    def in_channels(self) -> int:
        return self._blocks[0].in_channels
    
    @property
    def out_channels(self) -> int:
        return self._blocks[-1].out_channels
    
    @property
    def blocks(self) -> List[BlockConfig]:
        return self._blocks

# Configuration for decoder
class DecoderConfig:
    def __init__(self, cfg: dict, num_block: int, fdrop: float, sdrop: float) -> None:
        if cfg.blocks is None:
            raise ValueError('Missing blocks field in decoder config')
        elif type(cfg.blocks) != list:
            raise ValueError('Blocks field in decoder config must be a list')
        elif len(cfg.blocks) != num_block:
            raise ValueError('The number of decoder blocks must be equal to the number of encoder blocks')
        
        self._blocks: List[BlockConfig] = [None] * len(cfg.blocks)
        out_ch = -1
        for idx, bc in reversed(list(enumerate(cfg.blocks))):
            block = BlockConfig(bc, out_channels=out_ch, fdrop=fdrop, sdrop=sdrop)
            out_ch = block.in_channels
            self._blocks[idx] = block
            
    def to_dict(self) -> dict:
        d = {'in-channels': self.in_channels,
             'out-channels': self.out_channels}
        
        blocks = []
        for b in self._blocks:
            blocks.append(b.to_dict())
        
        d['blocks'] = blocks
        
        return d
            
    @property
    def in_channels(self) -> int:
        return self._blocks[0].in_channels
    
    @property
    def out_channels(self) -> int:
        return self._blocks[-1].out_channels
    
    @property
    def blocks(self) -> List[BlockConfig]:
        return self._blocks

# Configuration for a block
class BlockConfig:
    def __init__(self, cfg: dict, out_channels: int, fdrop: float, sdrop: float) -> None:
        # Set channels
        if cfg.channels is None:
            raise ValueError('Missing channels field in block config')
        elif type(cfg.channels) != int:
            raise ValueError('Channel field in block config must be an integer')
        
        self._in_channels = cfg.channels
        self._out_channels = out_channels if out_channels > 0 else cfg.channels
        
        # Set layers
        if cfg.layers is None:
            raise ValueError('Missing layers field in block config')
        elif type(cfg.layers) != list:
            raise ValueError('Layers field in block confif must be a list')
        
        self._layers: List[LayerConfig] = []
        in_ch = self.in_channels
        out_ch = self.in_channels
        for idx, lc in enumerate(cfg.layers):
            if idx == len(cfg.layers) - 1:
                out_ch = self.out_channels
            self._layers.append(LayerConfig(lc, in_ch, out_ch, fdrop, sdrop))
            
    def to_dict(self) -> dict:
        d = {'in-channels': self.in_channels,
             'out-channels': self.out_channels}
        
        layers = []
        for l in self._layers:
            layers.append(l.to_dict())
        
        d['layers'] = layers
        
        return d
    
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
    def __init__(self, cfg: dict, in_channels: int, out_channels: int, fdrop: float, sdrop: float) -> None:
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
            self._branches.append(BranchConfig(bc, in_channels, in_channels, fdrop, sdrop))
            
    def to_dict(self) -> dict:
        d = {'in-channels': self.in_channels,
             'out-channels': self.out_channels}
        
        branches = []
        for b in self._branches:
            branches.append(b.to_dict())
        
        d['branches'] = branches
        
        return d
            
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
    def __init__(self, cfg: dict, in_channels: int, out_channels: int, fdrop: float, sdrop: float) -> None:
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
        
        self._feature_dropout = fdrop
        self._sublayer_dropout = sdrop
        
    def to_dict(self) -> dict:
        d = {'in-channels': self._in_channels,
             'out-channels': self._out_channels, 
             'window': self._window,
             'dilation': self._dilation,
             'num-heads': self._num_heads}
        
        return d
        
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
    
    @property
    def feature_dropout(self) -> float:
        return self._feature_dropout
    
    @property
    def sublayer_dropout(self) -> float:
        return self._sublayer_dropout
