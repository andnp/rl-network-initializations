import torch
from collections import namedtuple

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

Batch = namedtuple(
    'batch',
    'states, nterm_next_states, actions, rewards, is_terminals, is_non_terminals, size'
)

def getBatchColumns(samples):
    s, a, sp, r, gamma = list(zip(*samples))
    states = torch.tensor(s, dtype=torch.float32, device=device)
    actions = torch.tensor(a, device=device).unsqueeze(1)
    rewards = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
    gamma = torch.tensor(gamma, device=device)

    is_terminal = gamma == 0

    sps = [x for x in sp if x is not None]
    if len(sps) > 0:

        non_final_next_states = torch.tensor(sps, dtype=torch.float32, device=device)
    else:
        non_final_next_states = torch.zeros((0, states.shape[1]))

    non_term = torch.logical_not(is_terminal).to(device)

    return Batch(states, non_final_next_states, actions, rewards, is_terminal, non_term, len(samples))

def toNP(maybeTensor):
    if type(maybeTensor) == torch.Tensor:
        return maybeTensor.cpu()

    return maybeTensor

def addGradients_(net1, net2):
    for (param1, param2) in zip(net1.parameters(), net2.parameters()):
        if param1.grad is not None and param2.grad is not None:
            param1.grad.add_(param2.grad)

def excludeParameters(param_list, exclude):
    exclude = [id(p) for layer in exclude for p in list(layer.parameters())]

    for param in param_list:
        if id(param) not in exclude:
            yield param

def getAllGrads(net):
    return [p.grad for p in net.parameters()]
