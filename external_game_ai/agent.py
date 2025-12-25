import torch


class Agent:
def __init__(self, policy_net):
self.policy_net = policy_net


def select_action(self, state_vector):
state = torch.tensor(state_vector, dtype=torch.float32)
probs = self.policy_net(state)
action = torch.multinomial(probs, 1).item()
return action
