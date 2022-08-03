##
##
##

from typing import Optional, Any
from torchmetrics import Metric
import torch

class CrossEntropyLoss(Metric):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: Optional[bool] = False
    
    def __init__(self, label_smoothing: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state('loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        self.loss += self.loss_fn(logits, target)
        self.count += 1
    
    def compute(self):
        return self.loss / self.count

class BCEWithLogitsLoss(Metric):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: Optional[bool] = False
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state('loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        self.loss += self.loss_fn(logits, target)
        self.count += 1
    
    def compute(self):
        return self.loss / self.count

class TotalLoss(Metric):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: Optional[bool] = False
    
    def __init__(self, recon_lambda: float, disc_lambda: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recon_lambda = recon_lambda
        self.disc_lambda = disc_lambda
        self.add_state('loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, recon_loss: torch.Tensor, disc_loss: torch.Tensor):
        self.loss += (self.recon_lambda * recon_loss) + (self.disc_lambda * disc_loss)
        self.count += 1
    
    def compute(self):
        return self.loss / self.count