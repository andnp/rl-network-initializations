from problems.MountainCar import MountainCar
from problems.CartPole import CartPole

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    if name == 'CartPole':
        return CartPole

    raise NotImplementedError()
