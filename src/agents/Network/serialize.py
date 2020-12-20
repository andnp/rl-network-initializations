from typing import Dict
import torch.nn as nn

def convOutputs(pixels: int, out_channels: int, kernel: int, stride: int):
    pixel_size = (pixels - (kernel - 1) - 1) // stride + 1

    return pixel_size**2 * out_channels

# json description -> pytorch layer
def deserializeLayer(layer_def: Dict, inputs: int):

    if layer_def['type'] == 'fc':
        units = layer_def['units']
        weights = nn.Linear(inputs, units, bias=layer_def.get('bias', True))
        outputs = units

    elif layer_def['type'] == 'conv':
        width = layer_def['width']
        out_channels = layer_def['out_channels']
        weights = nn.Conv2d(inputs, out_channels, kernel_size=3, stride=1)
        outputs = convOutputs(width, out_channels, 3, 1)

    else:
        raise NotImplementedError()

    activation = nn.Identity()

    act = layer_def.get('act', 'linear')
    if act == 'relu':
        activation = nn.ReLU()

    elif act == 'linear':
        activation = nn.Identity()

    else:
        raise NotImplementedError()

    return weights, activation, outputs
