##
##
##

from __future__ import annotations
from typing import List, Optional, Tuple
from enum import Enum

from .. import utils

# Configuration for model
class ModelConfig:
    def __init__(self, cfg: dict) -> None:
        self._name = cfg.name
        
        if cfg.architecture is None:
            raise ValueError('Missing architecture field in model config')
        model_cfg = utils.load_config_file(cfg.architecture)
        
        self._dropout = DropoutConfig(model_cfg.dropout)
        
        if model_cfg.encoder is None:
            raise ValueError('Missing encoder field in model config')
        self._encoder = EncoderConfig(model_cfg.encoder, self._dropout)
        
        if model_cfg.decoder is None:
            raise ValueError('Missing decoder field in model config')
        self._decoder = DecoderConfig(model_cfg.decoder, self._encoder, self._dropout)
        
        if model_cfg.embeddings is None:
            raise ValueError('Missing embeddings field in model config')
        self._embeddings = EmbeddingsConfig(model_cfg.embeddings, self._encoder)
        
    def to_dict(self, architecture: bool) -> dict:
        d = {'name': self._name }
        
        if architecture:
            d.update({
                'dropout': self._dropout.to_dict(),
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
    def dropout(self) -> DropoutConfig:
        return self._dropout

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
    def __init__(self, cfg: dict, dropout: DropoutConfig) -> None:
        if cfg.blocks is None:
            raise ValueError('Missing blocks field in encoder config')
        elif type(cfg.blocks) != list:
            raise ValueError('Blocks field in encoder config must be a list')
        
        self._blocks: List[BlockConfig] = [None] * len(cfg.blocks)
        out_ch = -1
        for idx, bc in reversed(list(enumerate(cfg.blocks))):
            bc.out_channels = out_ch
            
            block = BlockConfig(bc, True, idx == len(cfg.blocks) - 1, dropout)
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
    def __init__(self, cfg: dict, encoder: EncoderConfig, dropout: DropoutConfig) -> None:
        if cfg.blocks is None:
            raise ValueError('Missing blocks field in decoder config')
        elif type(cfg.blocks) != list:
            raise ValueError('Blocks field in decoder config must be a list')
        elif len(cfg.blocks) != len(encoder.blocks):
            raise ValueError('The number of decoder blocks must be equal to the number of encoder blocks')
        
        self._blocks: List[BlockConfig] = [None] * len(cfg.blocks)
        out_ch = -1
        for idx, bc in reversed(list(enumerate(cfg.blocks))):
            bc.out_channels = out_ch
            
            block = BlockConfig(bc, False, idx == len(cfg.blocks) - 1, dropout)
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
    def __init__(self, cfg: dict, encoder: bool, last: bool, dropout: DropoutConfig) -> None:
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
            
            if not last and idx == len(cfg.layers) - 1:
                sample = SampleType.DOWN if encoder else SampleType.UP
            else:
                sample = SampleType.NONE
            
            self._layers.append(LayerConfig(
                lc, in_ch, out_ch, sample, dropout))
            
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
    NONE = 'none'
    LARGER = 'larger'
    SMALLER = 'smaller'
    
class SampleType(Enum):
    UP = 0
    NONE = 1
    DOWN = 2

# Configuration for a layer
class LayerConfig:
    def __init__(self, 
                 cfg: dict, 
                 in_channels: int,
                 out_channels: int,
                 sample: SampleType, 
                 dropout: DropoutConfig) -> None:
        # Set channels
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._dropout = dropout
        
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
        
        #if self._in_channels % len(cfg.branches) != 0:
        #    raise ValueError(f'Number of branches in a layer must divide the number of channels of that layer')
        
        self._branches: List[LayerConfig] = []
        last_prod = -1
        branch_channels = self._in_channels // len(cfg.branches)
        for idx, bc in enumerate(cfg.branches):
            if (idx == 0 and self._cross_view == CrossViewType.LARGER) or \
                (idx == (len(cfg.branches) - 1) and self._cross_view == CrossViewType.SMALLER):
                cross_view = False
            else:
                cross_view = self._cross_view != CrossViewType.NONE
            
            branch = BranchConfig(
                bc,
                self._in_channels,
                self._in_channels,
                self._num_heads,
                cross_view,
                dropout)
            if branch.window * branch.dilation < last_prod:
                raise ValueError(f'Branches must be listed in ascending order of size') 
            else:
                last_prod = branch.window * branch.dilation
                self._branches.append(branch)
        
        if len(self._branches) == 1 and self._cross_view != CrossViewType.NONE:
            raise ValueError(f'A layer containing only one branch cannot be cross-view')
        
        # Set temporal
        if cfg.temporal is None:
            raise ValueError('Missing temporal field in layer config')
        elif type(cfg.temporal) != list:
            raise ValueError('Temporal field in layer confif must be a list')
        
        self._temporal = TemporalConfig(
            cfg.temporal, 
            sample,
            self._in_channels * len(self._branches), 
            self._out_channels,
            dropout)
            
    def to_dict(self) -> dict:
        d = {'cross-view': str(self._cross_view.value), 
             'num-heads': self._num_heads}
        
        branches = []
        for b in self._branches:
            branches.append(b.to_dict())
        
        d['branches'] = branches
        d['temporal'] = self._temporal.to_dict()
        
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
    
    @property
    def temporal(self) -> TemporalConfig:
        return self._temporal
    
    @property
    def dropout(self) -> DropoutConfig:
        return self._dropout

class BranchConfig:
    def __init__(self, 
                 cfg: dict,
                 in_channels: int,
                 channels: int,
                 num_heads: int,
                 cross_view: bool,
                 dropout: DropoutConfig) -> None:
        self._in_channels = in_channels
        self._channels = channels
        self._num_heads = num_heads
        self._cross_view = cross_view
        
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
        
        self._dropout = dropout
        
    def to_dict(self) -> dict:
        d = {'window': self._window,
             'dilation': self._dilation}
        
        return d
    
    @property
    def cross_view(self) -> bool:
        return self._cross_view
    
    @property
    def in_channels(self) -> int:
        return self._in_channels
        
    @property
    def channels(self) -> int:
        return self._channels
    
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
    def dropout(self) -> DropoutConfig:
        return self._dropout

class TemporalConfig:
    def __init__(self, 
                 cfg: dict, 
                 sample: SampleType, 
                 in_channels: int, 
                 out_channels: int, 
                 dropout: DropoutConfig) -> None:
        self._sample = sample
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._dropout = dropout
        
        self._branches = []
        for branch in cfg:
            if branch.window is None:
                raise ValueError('Missing window field in temporal config')
            elif type(branch.window) != int:
                raise ValueError('Window field in temporal config must be an integer')
            
            if branch.dilation is None:
                raise ValueError('Missing dilation field in temporal config')
            elif type(branch.dilation) != int:
                raise ValueError('Dilation field in temporal config must be an integer')
            
            self._branches.append((branch.window, branch.dilation))
        
        if self._out_channels % (len(self._branches) + 2) != 0:
            raise ValueError('Number of branches in temporal convolution is not valid')
        
    def to_dict(self) -> List[dict]:
        branches = []
        for (window, dilation) in self._branches:
            branches.append({'window': window, 'dilation': dilation})
        
        return branches
    
    @property
    def residual(self) -> bool:
        return True
    
    @property
    def sample(self) -> SampleType:
        return self._sample
    
    @property
    def branches(self) -> List[Tuple[int, int]]:
        return self._branches
    
    @property
    def in_channels(self) -> int:
        return self._in_channels
    
    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    @property
    def dropout(self) -> DropoutConfig:
        return self._dropout

class DropoutConfig:
    def __init__(self, cfg: Optional[dict]) -> None:
        if cfg is None:
            self._sublayer = 0.0
            self._feature = 0.0
            self._layer = 0.0
            self._head = DropHeadConfig(None)
        else:
            if cfg.sublayer is None:
                self._sublayer = 0.0
            else:
                self._sublayer = float(cfg.sublayer)
                
            if cfg.feature is None:
                self._feature = 0.0
            else:
                self._feature = float(cfg.feature)
            
            if cfg.layer is None:
                self._layer = 0.0
            else:
                self._layer = float(cfg.layer)
            
            self._head = DropHeadConfig(cfg.head)
    
    def to_dict(self) -> dict:
        d = {'sublayer': self._sublayer,
             'feature': self._feature,
             'layer': self._layer,
             'head': self._head.to_dict()}
        
        return d
    
    @property
    def sublayer(self) -> float:
        return self._sublayer
    
    @property
    def feature(self) -> float:
        return self._feature
    
    @property
    def layer(self) -> float:
        return self._layer
    
    @property
    def head(self) -> DropHeadConfig:
        return self._head
    
class DropHeadConfig:
    def __init__(self, cfg: Optional[dict]) -> None:
        if cfg is None:
            self._start = 0.0
            self._end = 0.0
            self._warmup = 0
        else:
            self._start =  float(cfg.start)
            self._end = float(cfg.end)
            self._warmup = int(cfg.warmup)
        
    def to_dict(self) -> dict:
        d = {
            'start': self._start,
            'end': self._end,
            'warmup': self._warmup
        }
        
        return d
    
    @property
    def start(self) -> float:
        return self._start
    
    @property
    def end(self) -> float:
        return self._end
    
    @property
    def warmup(self) -> int:
        return self._warmup