##
##
##

from __future__ import annotations
from typing import List
from enum import Enum

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
        
        if model_cfg.structure_dropout is not None:
            self._structure_dropout_start = float(model_cfg.structure_dropout.start)
            self._structure_dropout_end = float(model_cfg.structure_dropout.end)
            self._structure_dropout_warmup = int(model_cfg.structure_dropout.warmup)
            
        self._structure_dropout = model_cfg.structure_dropout
        if self._structure_dropout is None:
            self._structure_dropout = 0.0
        
        if model_cfg.encoder is None:
            raise ValueError('Missing encoder field in model config')
        model_cfg.encoder.feature_dropout = self._feature_dropout
        model_cfg.encoder.sublayer_dropout = self._sublayer_dropout
        self._encoder = EncoderConfig(model_cfg.encoder)
        
        if model_cfg.decoder is None:
            raise ValueError('Missing decoder field in model config')
        model_cfg.decoder.feature_dropout = self._feature_dropout
        model_cfg.decoder.sublayer_dropout = self._sublayer_dropout
        self._decoder = DecoderConfig(model_cfg.decoder, self._encoder)
        
        if model_cfg.embeddings is None:
            raise ValueError('Missing embeddings field in model config')
        self._embeddings = EmbeddingsConfig(model_cfg.embeddings, self._encoder)
        
    def to_dict(self, architecture: bool) -> dict:
        d = {'name': self._name }
        
        if architecture:
            d.update({
                'sublayer_dropout': self._sublayer_dropout,
                'feature_dropout': self._feature_dropout,
                'structure_dropout': self._structure_dropout,
                'embeddings': self._embeddings.to_dict(),
                'encoder': self._encoder.to_dict(),
                'decoder': self._decoder.to_dict()
            })
        
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
    def sublayer_dropout(self) -> float:
        return self._sublayer_dropout
    
    @property
    def feature_dropout(self) -> float:
        return self._feature_dropout
    
    @property
    def structure_dropout_start(self) -> float:
        return self._structure_dropout_start
    
    @property
    def structure_dropout_end(self) -> float:
        return self._structure_dropout_end
    
    @property
    def structure_dropout_warmup(self) -> int:
        return self._structure_dropout_warmup

# Configuration for embeddings
class EmbeddingsConfig:
    def __init__(self, cfg: dict, encoder: EncoderConfig) -> None:
        self._out_channels = encoder.in_channels
        
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
    
    def to_dict(self) -> dict:
        d = {'temporal-channels': self._temporal_channels, 
             'type-channels': self._type_channels}
        
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

# Configuration for encoder
class EncoderConfig:
    def __init__(self, cfg: dict) -> None:
        if cfg.blocks is None:
            raise ValueError('Missing blocks field in encoder config')
        elif type(cfg.blocks) != list:
            raise ValueError('Blocks field in encoder config must be a list')
        
        self._blocks: List[BlockConfig] = [None] * len(cfg.blocks)
        out_ch = -1
        for idx, bc in reversed(list(enumerate(cfg.blocks))):
            bc.feature_dropout = cfg.feature_dropout
            bc.sublayer_dropout = cfg.sublayer_dropout
            bc.out_channels = out_ch
            
            block = BlockConfig(bc)
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
    def __init__(self, cfg: dict, encoder: EncoderConfig) -> None:
        if cfg.blocks is None:
            raise ValueError('Missing blocks field in decoder config')
        elif type(cfg.blocks) != list:
            raise ValueError('Blocks field in decoder config must be a list')
        elif len(cfg.blocks) != len(encoder.blocks):
            raise ValueError('The number of decoder blocks must be equal to the number of encoder blocks')
        
        self._blocks: List[BlockConfig] = [None] * len(cfg.blocks)
        out_ch = -1
        for idx, bc in reversed(list(enumerate(cfg.blocks))):
            bc.feature_dropout = cfg.feature_dropout
            bc.sublayer_dropout = cfg.sublayer_dropout
            bc.out_channels = out_ch
            
            block = BlockConfig(bc)
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
    def __init__(self, cfg: dict) -> None:
        # Set channels
        if cfg.channels is None:
            raise ValueError('Missing channels field in block config')
        elif type(cfg.channels) != int:
            raise ValueError('Channel field in block config must be an integer')
        
        self._in_channels = cfg.channels
        self._out_channels = cfg.out_channels if cfg.out_channels > 0 else cfg.channels
        
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
            lc.in_channels = in_ch
            lc.out_channels = out_ch
            lc.feature_dropout = cfg.feature_dropout
            lc.sublayer_dropout = cfg.sublayer_dropout
            self._layers.append(LayerConfig(lc))
            
    def to_dict(self) -> dict:
        d = {'channels': self._in_channels}
        
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

class CrossViewType(Enum):
    NONE = 'none',
    LARGER = 'larger',
    SMALLER = 'smaller',

# Configuration for a layer
class LayerConfig:
    def __init__(self, cfg: dict) -> None:
        # Set channels
        self._in_channels = cfg.in_channels
        self._out_channels = cfg.out_channels
        
        # Set cross view
        if cfg.cross_view is None:
            self._cross_view = CrossViewType.NONE
        else:
            self._cross_view = CrossViewType[cfg.cross_view.upper()]
        
        self._num_heads = cfg.num_heads
        if self._num_heads is None:
            raise ValueError('Missing num-heads field in layer config')
        elif type(self._num_heads) != int:
            raise ValueError('Num-heads field in layer config must be an integer')
        elif self._in_channels % self._num_heads != 0:
            raise ValueError('Number of heads in a layer must divide the number of input channels')
        
        # Set branches
        if cfg.branches is None:
            raise ValueError('Missing branches field in layer config')
        elif type(cfg.branches) != list:
            raise ValueError('Branches field in layer confif must be a list')
        
        if self._out_channels % len(cfg.branches) != 0:
            raise ValueError(f'Number of branches in a layer must divide the number of channels of that layer')
        
        self._branches: List[LayerConfig] = []
        last_prod = -1
        for idx, bc in enumerate(cfg.branches):
            bc.channels = self._in_channels
            bc.out_channels = self._out_channels // len(cfg.branches)
            bc.num_heads = self._num_heads
            bc.feature_dropout = cfg.feature_dropout
            bc.sublayer_dropout = cfg.sublayer_dropout
            if (idx == 0 and self._cross_view == CrossViewType.LARGER) or \
                (idx == (len(cfg.branches) - 1) and self._cross_view == CrossViewType.SMALLER):
                bc.cross_view = False
            else:
                bc.cross_view = self._cross_view != CrossViewType.NONE
            
            branch = BranchConfig(bc)
            if branch.window * branch.dilation < last_prod:
                raise ValueError(f'Branches must be listed in ascending order of size') 
            else:
                last_prod = branch.window * branch.dilation
                self._branches.append(branch)
        
        if len(self._branches) == 1 and self._cross_view != CrossViewType.NONE:
            raise ValueError(f'A layer containing only one branch cannot be cross-view')
            
    def to_dict(self) -> dict:
        d = {'cross-view': str(self._cross_view), 
             'num-heads': self._num_heads}
        
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
    def cross_view(self) -> CrossViewType:
        return self._cross_view
    
    @property
    def branches(self) -> List[BranchConfig]:
        return self._branches

class BranchConfig:
    def __init__(self, cfg: dict) -> None:
        self._channels = cfg.channels
        self._out_channels = cfg.out_channels
        self._num_heads = cfg.num_heads
        self._cross_view = cfg.cross_view
        
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
        
        self._feature_dropout = cfg.feature_dropout
        self._sublayer_dropout = cfg.sublayer_dropout
        
    def to_dict(self) -> dict:
        d = {'window': self._window,
             'dilation': self._dilation}
        
        return d
    
    @property
    def cross_view(self) -> bool:
        return self._cross_view
        
    @property
    def channels(self) -> int:
        return self._channels
    
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
    
    @property
    def structure_dropout(self)-> float:
        return 0.0 # it will be overwritten by the scheduler
