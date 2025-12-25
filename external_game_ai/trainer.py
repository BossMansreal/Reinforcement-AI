import torch
import torch.optim as optim


class Trainer:
def __init__(self, policy_net, lr):
self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
self.policy_net = policy_net


def update(self, memory):
discounted_rewards = []
cumulative = 0


for r in reversed(memory.rewards):
cumulative = r + 0.99 * cumulative
discounted_rewards.insert(0, cumulative)


rewards = torch.tensor(discounted_rewards)
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)


loss = 0
for state, action, reward in zip(memory.states, memory.actions, rewards):
state = torch.tensor(state, dtype=torch.float32)
probs = self.policy_net(state)
loss += -torch.log(probs[action]) * reward


self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
