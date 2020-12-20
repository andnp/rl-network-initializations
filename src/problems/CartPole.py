from problems.BaseProblem import BaseProblem
from environments.Gym import Gym
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class CartPole(BaseProblem):
    def __init__(self, exp, idx, seed):
        super().__init__(exp, idx, seed)
        self.actions = 2

        self.rep = IdentityRep()

        self.features = 4
        self.gamma = 0.99

        # trick gym into thinking the max number of steps is a bit longer
        # that way we get to control the termination at max steps
        self.env = Gym('CartPole-v1', self.seed, None)
