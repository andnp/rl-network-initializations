from typing import Dict
import torch

from agents.BaseAgent import BaseAgent
from agents.Network.Network import Network
from utils.torch import device

class NonlinearQ(BaseAgent):
    def __init__(self, features: int, actions: int, params: Dict, seed: int, init_seed: int):
        super().__init__(features, actions, params, seed)

        # set up initialization of the value function network
        # and target network
        self.value_net = Network(features, actions, params, init_seed).to(device)

    def values(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        return self.value_net(x)[0].detach().cpu().numpy()

    def update(self, s, a, sp, r, gamma):
        pass
