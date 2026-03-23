import logging
from typing import List
import torch.nn as nn
from stefutil.prettier import get_logger, style as s


__all__ = ['ValueFunction']


_logger = get_logger('Value-Function')


class ValueFunction(nn.Module):
    def __init__(self, input_dim: int = None, hidden_dims: List[int] = None, num_attributes: int = 5, logger: logging.Logger = None):
        super(ValueFunction, self).__init__()
        layer_dims = [input_dim] + hidden_dims + [num_attributes]
        # use a sequential module, linear layers followed by relu
        layers = []
        n_dim = len(layer_dims)
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_dim - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        (_logger or logger).info(f'ValueFunction initialized w/ layer dimensions: {s.i(layer_dims)} and {self}')

    def forward(self, x):
        x = self.layers(x)
        # x = torch.sigmoid(x)*4
        return x
