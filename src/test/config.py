##
##
##

from typing import List

from src.dataset.config import DatasetConfig
from src.model.config import ModelConfig

class TestConfig:
    def __init__(self,
                 options: dict, 
                 datasets: List[DatasetConfig],
                 models: List[ModelConfig]) -> None:
        pass