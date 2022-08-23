##
##
##

from __future__ import annotations
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .norm import LayerNorm2d
from .config import BlockConfig, CrossViewType, LayerConfig, BranchConfig, SampleType, TemporalConfig
from ..dataset.skeleton import SkeletonGraph

class Block(nn.Module):
    
    def __init__(self, config: BlockConfig,skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        for layer_cfg in config.layers:
            self.layers.append(Layer(layer_cfg, skeleton))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            out = layer(x)
            x = out 
        
        return out

class Layer(nn.Module):
    def __init__(self, config: LayerConfig, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self._cross_view = config.cross_view
        if config.in_channels != config.out_channels:
            self.residual = nn.Conv2d(config.in_channels, config.out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

        self.branches = nn.ModuleList()
        out_branch = config.out_channels // len(config.branches)
        for branch_cfg in config.branches:
            if config.in_channels != branch_cfg.channels:
                down = nn.Conv2d(config.in_channels, branch_cfg.channels, kernel_size=1)
            else:
                down = nn.Identity()
                
            if out_branch != branch_cfg.channels:
                up = nn.Conv2d(branch_cfg.channels, out_branch, kernel_size=1)
            else:
                up = nn.Identity()
            
            self.branches.append(nn.ModuleList([down, Branch(branch_cfg, skeleton), up]))
            
        self._layer_drop = config.dropout.layer
    
    def _larger(self, x: torch.Tensor) -> torch.Tensor:
        output = []
        cross_x = None 
        for branch in self.branches:
            red_x = branch[0](x)
            out = branch[1](red_x, cross_x)
            output.append(branch[2](out))
            
            if self._cross_view == CrossViewType.LARGER:
                cross_x = out
        
        output = torch.cat(output, dim=1)
        return output
    
    def _smaller(self, x: torch.Tensor) -> torch.Tensor:
        output = []
        cross_x = None 
        for branch in reversed(self.branches):
            red_x = branch[0](x)
            out = branch[1](red_x, cross_x)
            output.append(branch[2](out))
            
            if self._cross_view == CrossViewType.SMALLER:
                cross_x = out
        
        output = torch.cat(output, dim=1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = 0
        
        prob = torch.tensor(1 - self._layer_drop)
        keep = torch.bernoulli(prob) == 1
        
        if keep:
            match self._cross_view:
                case CrossViewType.NONE:
                    out = self._larger(x)
                case CrossViewType.LARGER:
                    out = self._larger(x)
                case CrossViewType.SMALLER:
                    out = self._smaller(x)
                case _:
                    raise ValueError('Unknown cross-view type')
        
        out += res
        
        return out

class Branch(nn.Module):
    def __init__(self, config: BranchConfig, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self.window = config.window
        padding = (config.window + (config.window - 1) * (config.dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(config.window, 1),
            dilation=(config.dilation, 1),
            padding=(padding, 0))
        
        # Attention
        self.first_norm = LayerNorm2d(config.channels)
        self.attention = SpatioTemporalAttention(config, skeleton)
        self.aggregate = nn.Conv2d(config.window * config.channels, config.channels, kernel_size=1)
        
        if config.cross_view:
            self.second_norm = LayerNorm2d(config.channels)
            self.cross_norm = LayerNorm2d(config.channels)
            self.cross_attention = SpatioTemporalAttention(config, skeleton)
            self.cross_aggregate = nn.Conv2d(config.window * config.channels, config.channels, kernel_size=1)
        
        # FFN
        self.ffn_norm = LayerNorm2d(config.channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, kernel_size=1),
            nn.GELU(), 
            nn.Dropout(config.dropout.feature),
            nn.Conv2d(config.channels, config.channels, kernel_size=1)
        )
        
        self.sublayer_dropout = nn.Dropout(config.dropout.sublayer)
    
    def _group_window(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        #(N, C * window, T, V)
        x = self.unfold(x)
        # (N, C, window, T, V)
        x = x.view(N, C, self.window, T, V)
        # (N, C, T, window, V)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        # (N, C, T, window * V)
        x = x.view(N, C, T, self.window * V)
        
        return x
    
    def _aggregate_window(self, x: torch.Tensor, cross: bool):
        N, C, T, _ = x.shape
        # (N, C, T, window, V)
        x = x.view(N, C, T, self.window, -1)
        # (N, C, window, T, V)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        # (N, C * window, T, V)
        x = x.view(N, C * self.window, T, -1)
        
        # (N, C, T, V)
        if cross:
            x = self.cross_aggregate(x)
        else:
            x = self.aggregate(x)
            
        return x
    
    def forward(self, x: torch.Tensor, cross_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention
        window_x = self._group_window(x)
        norm_x = self.first_norm(window_x)
        att_out = self.attention(norm_x, norm_x)
        att_out = self._aggregate_window(att_out, False)
        att_out = self.sublayer_dropout(att_out)
        att_out += x
        
        # Cross Attention
        if cross_x is not None:
            x_q = self.second_norm(self._group_window(att_out))
            x_kv = self.cross_norm(self._group_window(cross_x))
            cross_att_out = self.cross_attention(x_q, x_kv)
            cross_att_out = self._aggregate_window(cross_att_out, True)
            cross_att_out = self.sublayer_dropout(cross_att_out)
            cross_att_out += att_out
        else:
            cross_att_out = att_out
        
        # FFN
        ffn_out = self.ffn_norm(cross_att_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.sublayer_dropout(ffn_out)
        ffn_out += cross_att_out

        return ffn_out

class SpatioTemporalAttention(nn.Module):
    
    def __init__(self, config: BranchConfig, _: SkeletonGraph) -> None:
        super().__init__()

        self._num_heads = config.num_heads
        self._channels = config.channels
        self._head_channels = config.channels // config.num_heads
        self._drop_head = config.dropout.head.start
        self.feature_dropout = nn.Dropout(config.dropout.feature)
        
        # Layers
        self.q_proj = nn.Conv2d(self._channels, self._channels, kernel_size=1)
        self.kv_proj = nn.Conv2d(self._channels, 2 * self._channels, kernel_size=1)
        self.o_proj = nn.Conv2d(self._channels, self._channels, kernel_size=1)
    
    @property
    def drop_head(self) -> float:
        return self._drop_head
    
    @drop_head.setter
    def drop_head(self, drop: float):
        self._drop_head = drop
    
    def _scaled_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(q.size(-1))
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.feature_dropout(attn)
        
        # (N, num_heads, L, C_head)
        values = torch.matmul(attn, v)
        
        return values
    
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        N, _, T, V = x_q.shape
        
        q: torch.Tensor = self.q_proj(x_q)
        q = q.view(N, self._num_heads, self._head_channels, T, V)
        # (N, num_heads, T, V, C_head)
        q = q.permute(0, 1, 3, 4, 2).contiguous()
        
        kv: torch.Tensor = self.kv_proj(x_kv)
        kv = kv.view(N, self._num_heads, 2 * self._head_channels, T, V)
        # (N, num_heads, T, V, C_head)
        kv = kv.permute(0, 1, 3, 4, 2).contiguous()
        k, v = torch.chunk(kv, chunks=2, dim=-1)
        
        # (N, num_heads, T, V, C_head)
        values: torch.Tensor = checkpoint(self._scaled_dot_product, q, k, v)
        
        if self.training:
            # (N, num_heads)
            prob = torch.tensor([[1 - self._drop_head]], device=values.device).expand(values.shape[:2])
            # (N, num_heads)
            epsilon = torch.bernoulli(prob)
            # (N, num_heads, 1, 1, 1)
            mask = epsilon.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            values = values * mask
            
        values = values.permute(0, 1, 4, 2, 3).contiguous()
        values = values.view(N, self._channels, T, V)
        output = self.o_proj(values)
        
        if self.training:
            # (N)
            factor = torch.sum(epsilon, dim=-1)
            # (N, 1, 1, 1)
            factor = factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            factor = factor / self._num_heads
            factor[factor == 0] = 1
            # (N, L, C_out)
            output = output / factor
        
        return output
   
class TemporalConvolution(nn.Module):
    def __init__(self, config: TemporalConfig) -> None:
        super().__init__()
        
        self._layer_drop = config.dropout.layer
        
        if config.sample == SampleType.UP:
            raise NotImplemented
        
        stride = 1 if config.sample == SampleType.NONE else 2
        
        self.branches = nn.ModuleList()
        branch_channels = config.out_channels // (len(config.branches) + 2)
        for (window, dilation) in config.branches:
            pad = (window + (window - 1) * (dilation - 1) - 1) // 2
            self.branches.append(nn.Sequential(
                LayerNorm2d(config.in_channels),
                nn.Conv2d(config.in_channels, branch_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout(config.dropout.feature),
                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=(window, 1),
                    stride=(stride, 1),
                    padding=(pad, 0),
                    dilation=(dilation, 1)
                ),
            ))
        
        # Add MaxPool
        self.branches.append(nn.Sequential(
            LayerNorm2d(config.in_channels),
            nn.Conv2d(config.in_channels, branch_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(config.dropout.feature),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
        ))
        
        # Add conv1x1
        self.branches.append(nn.Sequential(
            LayerNorm2d(config.in_channels),
            nn.Conv2d(config.in_channels, branch_channels, kernel_size=1, stride=(stride, 1)),
        ))
        
        if not config.residual:
            self.residual = lambda _: 0
        elif (config.in_channels == config.out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(
                config.in_channels, 
                config.out_channels, 
                kernel_size=1, 
                stride=(stride, 1))
            
        self.sublayer_dropout = nn.Dropout(config.dropout.sublayer)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        
        outputs = []
        for branch in self.branches:
            out = branch(x)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = self.sublayer_dropout(out)
        
        out += res
        
        return out