from abc import abstractmethod
from typing import Dict
import numpy as np

from utils.policies import buildEGreedyPolicy

class BaseAgent:
    def __init__(self, features: int, actions: int, params: Dict, seed: int):
        self.features = features
        self.actions = actions
        self.params = params
        self.policy_rng = np.random.RandomState(seed)
        self.seed = seed

        # define parameter contract
        self.epsilon: float = params['epsilon']

        # have the agent build the representation
        # but let the "agent_wrapper" deal with using it
        # that way we can save a bit of compute by caching things
        self.rep_params: Dict = params.get('representation', { 'type': 'scale' })

        # learnable parameters
        self.w: np.ndarray = np.zeros((actions, self.features))

        # create a policy utility object to help keep track of sampling from the probabilities
        self.policy = buildEGreedyPolicy(self.policy_rng, self.epsilon, self.values)

    # compute the value function given a numpy input
    # returns an np.array of size <actions>
    def values(self, x: np.ndarray):
        return self.w.dot(x)

    # just to conform to rlglue interface
    # passes action selection on to the policy utility object
    def selectAction(self, x: np.ndarray):
        return self.policy.selectAction(x)

    # where the learning magic happens
    # uses state-based gamma (so no need to handle terminal states specially)
    @abstractmethod
    def update(self, x: np.ndarray, a: int, xp: np.ndarray, r: float, gamma: float):
        pass

    def cleanup(self):
        pass
