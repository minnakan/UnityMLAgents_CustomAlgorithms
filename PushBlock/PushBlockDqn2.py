import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, log_dir="runs/experiment"):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_every = 4
        self.t_step = 0
        self.writer = SummaryWriter(log_dir)  # Initialize TensorBoard writer with log directory

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.numpy())

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        # Log the loss and epsilon to TensorBoard
        self.writer.add_scalar('Loss', loss.item(), self.t_step)
        self.writer.add_scalar('Epsilon', self.epsilon, self.t_step)

    def soft_update(self, local_model, target_model, tau=0.001):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# Initialize the environment and agent
env = UnityEnvironment(file_name="PushBlock")
env.reset()
behavior_name = list(env.behavior_specs)[0]
state_size = env.behavior_specs[behavior_name].observation_specs[0].shape[0]
action_size = env.behavior_specs[behavior_name].action_spec.discrete_branches[0]

# Placeholder for TensorBoard log directory
log_dir = "C:/Users/minna/Documents/Repository/tensorboard/PushBlock/"
agent = DQNAgent(state_size, action_size, learning_rate=0.001, log_dir=log_dir)

# Training loop
for episode in range(2000):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    state = decision_steps.obs[0][0]

    # Update target network every 50 episodes
    if(episode % 50 == 0):
        agent.soft_update(agent.qnetwork_local, agent.qnetwork_target,1.0)

    episode_reward = 0
    for t in range(1000):
        action = agent.act(state)
        action_tuple = ActionTuple(discrete=np.array([[action]]))
        env.set_actions(behavior_name, action_tuple)
        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(decision_steps) > 0:
            next_state = decision_steps.obs[0][0]
            reward = decision_steps.reward[0]
        else:
            next_state = state
            reward = terminal_steps.reward[0]

        # Apply the step penalty
        reward += -0.0025

        done = len(terminal_steps) > 0

        # Add reward for pushing the block
        if reward == 1.0:
            reward += 1.0  # Block touches goal reward

        agent.step(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward


        if done:
            break

    # Log episode reward to TensorBoard
    agent.writer.add_scalar('Episode Reward', episode_reward, episode)
    print(f"Episode {episode} finished with reward {episode_reward}")

env.close()
agent.writer.close()  # Close the TensorBoard writer
