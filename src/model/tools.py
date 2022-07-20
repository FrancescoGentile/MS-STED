##
##
##

from torch import nn

def init_layers(layer: nn.Module):
    if isinstance(layer, nn.Conv2d):
        if layer.weight is not None:
            nn.init.kaiming_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, mean=0, std=0.01)
    elif isinstance(layer, nn.BatchNorm2d):
        if layer.weight is not None:
            nn.init.constant_(layer.weight, 1)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 1e-3)
    '''
    elif isinstance(layer, nn.Linear):
        if layer.weight is not None:
            nn.init.kaiming_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, mean=0, std=0.01)
    '''