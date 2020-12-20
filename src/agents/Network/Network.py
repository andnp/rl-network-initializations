from typing import Any, Dict, List
import torch
import torch.nn as nn

from agents.Network.serialize import deserializeLayer

def isLastConv(layer_defs: List[Any], i: int):
    t = layer_defs[i]['type']

    n = None
    if i < len(layer_defs) - 1:
        n = layer_defs[i + 1]['type']

    return t == 'conv' and n != 'conv'

class Network(nn.Module):
    def __init__(self, inputs: int, outputs: int, params: Dict[str, Any], seed: int):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.seed = seed

        torch.manual_seed(seed)

        self.model = nn.Sequential()
        layer_defs = params.get('layers', [])
        for i, layer_def in enumerate(layer_defs):
            # takes a json description of a neural network layer
            # and returns torch parameters, activation function, and the number of outputs (which will be the number of inputs for the next layer)
            weights, activation, inputs = deserializeLayer(layer_def, inputs)

            self.model.add_module(f'layer-{i}-weights', weights)
            self.model.add_module(f'layer-{i}-activation', activation)

            # initialize full connected layers with xavier init
            if layer_def['type'] == 'fc':
                gain = nn.init.calculate_gain(layer_def['act'])
                nn.init.xavier_uniform_(weights.weight, gain)

            if weights.bias is not None:
                nn.init.normal_(weights.bias, 0, 0.1)

            # if there are no more conv layers, go ahead and flatten for follow-up fully-connected layers
            if isLastConv(layer_defs, i):
                self.model.add_module(f'layer-{i}-flatten', nn.Flatten())


        self.features = inputs
        self.output = nn.Linear(inputs, outputs, bias=params.get('output_bias', True))
        self.output_layers = [self.output]
        self.output_grads = [True]

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        outs = []
        for layer, grad in zip(self.output_layers, self.output_grads):
            if grad:
                outs.append(layer(x))
            else:
                outs.append(layer(x.detach()))

        return outs
