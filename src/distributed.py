##
##
##

from enum import Enum
import os
import ipaddress
from typing import Optional, Tuple
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import utils

class DeviceType(Enum):
    GPU = 'gpu'
    CPU = 'cpu'
    
class Backend(Enum):
    GPU = 'nccl'
    CPU = 'gloo'

class DistributedConfig:
    def __init__(self, path: Optional[str], rank: Optional[int]) -> None:
        if path is not None:
            if rank is None:
                raise ValueError(f'You need to specify the rank for this node')
            self._set_from_file(path, rank)
        else:
            self._set_default()
    
    def _set_default(self):
        if torch.cuda.is_available():
            self._world_size = torch.cuda.device_count()
            self._local_world_size = self._world_size
            self._group_rank = 0
            self._group_start = 0
            self._master_addr = '127.0.0.1'
            self._master_port = '24900'
            self._gpu_ids = list(range(self._world_size))
            self._backend = Backend.GPU
        else:
            self._world_size = 1
            self._local_world_size = 1
            self._group_rank = 0
            self._group_start = 0
            self._master_addr = '127.0.0.1'
            self._master_port = '24900'
            self._gpu_ids = [None]
            self._backend = Backend.CPU
    
    def _set_from_file(self, path: str, this_rank: int):
        cfg = utils.load_config_file(path)
        if cfg.nodes is None:
            raise ValueError('Missing nodes field in distributed config')
        elif type(cfg.nodes) != list:
            raise ValueError('Nodes field in distributed config must be a list')
                
        nodes = [None]
        has_cpu = False
        ranks = {}
        self._world_size = 0
        this_pos = None
        for ncfg in cfg.nodes:
            node, cpu = self._parse_node(ncfg)
            has_cpu = has_cpu or cpu
            # Check that rank is unique
            rank = node['rank']
            if ranks.get(rank) is not None:
                raise ValueError(f'rank {rank} assigned to more than one node')
            else:
                ranks.update({rank: True})
            
            if node['master']:
                if nodes[0] is not None:
                    raise ValueError(f'Only one node can be a master')
                else:
                    nodes[0] = node
                    self._master_addr = node['address']
                    self._master_port = node['port']
                    pos = 0
            else:
                nodes.append(node)
                pos = len(node) - 1
            
            devices = node['devices']
            self._world_size += len(devices)
            if rank == this_rank:
                self._local_world_size = len(devices)
                self._group_rank = pos
                this_pos = pos
                self._gpu_ids = devices
                
                for d in devices:
                    if d is not None and d >= torch.cuda.device_count():
                        raise ValueError(f'gpu {d} does not exist on this node')
        
        if this_pos is None:
            raise ValueError('This node was not found in the list of nodes')
        
        self._group_start = 0
        for i in range(this_pos):
            self._group_start += len(nodes[i]['devices'])
            
        self._backend = Backend.CPU if has_cpu else Backend.GPU
        
    def _parse_node(self, cfg: dict) -> Tuple[dict, bool]:
        node = {}
        
        node['rank'] = cfg.rank
        if cfg.rank is None:
            raise ValueError('Missing rank field in node config')
        elif type(cfg.rank) != int or cfg.rank < 0:
            raise ValueError('Rank field in node config must be a non negative integer')
        
        if cfg.master is not None:
            node['master'] = True
            if cfg.master.address is None:
                raise ValueError('Missing address field in master config')
            node['address'] = str(ipaddress.ip_address(cfg.master.address))
            if cfg.master.port is None:
                node['port'] = '29400'
            else:
                node['port'] = str(cfg.master.port)

        if cfg.devices is None:
            raise ValueError('Missing devices field in node config')
        
        devices = []
        has_cpu = False
        if cfg.devices.gpus is not None:
            if type(cfg.devices.gpus) == int:
                devices += list(range(cfg.devices.gpus))
            elif type(cfg.devices.gpus) == list:
                devices += cfg.devices.gpus
            else:
                raise ValueError('gpus field in devices config must be an integer or a list')
        
        if cfg.devices.cpus is not None:
            if type(cfg.devices.cpus) == int:
                has_cpu = True
                devices += [None for _ in range(cfg.devices.cpus)]
            else:
                raise ValueError('cpus field in devices config must be an integer')
        
        node['devices'] = devices
        
        if len(devices) == 0:
            raise ValueError(f'No device was specified for node with rank {cfg.rank}')
                    
        return node, has_cpu
    
    @property 
    def world_size(self) -> int:
        return self._world_size
    
    @property
    def local_world_size(self) -> int:
        return self._local_world_size
    
    @property
    def group_rank(self) -> int:
        return self._group_rank
    
    @property
    def rank(self) -> int:
        return self._rank
    
    @property
    def local_rank(self) -> int:
        return self._local_rank
    
    @local_rank.setter
    def local_rank(self, local_rank: int):
        self._local_rank = local_rank
        self._rank = self._group_start + local_rank
    
    @property
    def master_addr(self) -> str:
        return self._master_addr
    
    @property
    def master_port(self) -> str:
        return self._master_port
    
    @property
    def backend(self) -> str:
        return self._backend.value
    
    def get_gpu_id(self) -> Optional[int]:
        return self._gpu_ids[self._local_rank]
    
    def is_local_master(self) -> bool:
        return self._local_rank == 0
    
    def is_master(self) -> bool:
        return self._rank == 0


def setup(local_rank: int, config: DistributedConfig, callback, *args):
    config.local_rank = local_rank
    
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
                
    dist.init_process_group(
        backend=config.backend, 
        rank=config.rank, 
        world_size=config.world_size)
    
    callback(*args)


def run(config: DistributedConfig, callback, *args):
    if config.local_world_size > 1:
        mp.spawn(
            fn=setup, 
            args=(config, callback, *args,), 
            nprocs=config.local_world_size, 
            join=True)
    else:
        setup(0, config, callback, *args)