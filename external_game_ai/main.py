from config import *
from network import PolicyNetwork
from agent import Agent
from trainer import Trainer
from memory import EpisodeMemory
from rewards import compute_reward


policy = PolicyNetwork(STATE_SIZE, ACTION_SIZE)
agent = Agent(policy)
trainer = Trainer(policy, LEARNING_RATE)
memory = EpisodeMemory()


previous_state = None


while True:
state = get_state_from_environment() # external
action = agent.select_action(state)
send_action_to_environment(action)


if previous_state is not None:
reward, done = get_reward_and_done()
if TRAINING:
memory.store(previous_state, action, reward)


if done:
if TRAINING:
trainer.update(memory)
memory.clear()


previous_state = state
