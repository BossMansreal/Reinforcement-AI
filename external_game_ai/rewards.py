def compute_reward(prev_state, current_state, terminated):
reward = 0.01 # survival incentive


if current_state[0] < prev_state[0]: # health dropped
reward -= 0.2


if terminated:
reward -= 5.0


return reward
