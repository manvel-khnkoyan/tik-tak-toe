# agent.py
import random
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    """Deep Q-Network model"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Learning agent"""
    def __init__(self, 
                 input_size=9, 
                 hidden_size=256, 
                 output_size=9,
                 alpha=0.001,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 batch_size=64,
                 buffer_size=10000,
                 target_update=10):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Initialize networks
        self.main_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=alpha, weight_decay=0.0001)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)
    
    def get_available_actions(self, board):
        return [tuple(pos) for pos in zip(*np.nonzero(board == 0))]
    
    def choose_action(self, board):
        available_actions = self.get_available_actions(board)
        if not available_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        state = torch.FloatTensor(board.flatten()).to(self.device)
        with torch.no_grad():
            q_values = self.main_net(state)
        
        # Convert available actions to indices
        action_indices = [row*3 + col for (row, col) in available_actions]
        best_idx = torch.argmax(q_values[action_indices]).item()
        return available_actions[best_idx]
    
    def store_experience(self, state, action, reward, next_state, done):
        if action is None:
            return
        action_idx = action[0]*3 + action[1]
        self.buffer.add(
            state.flatten(),
            action_idx,
            reward,
            next_state.flatten(),
            done
        )
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.main_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states)
            # Mask invalid moves
            mask = (next_states == 0).float()
            masked_next_q = next_q * mask + (1 - mask) * -1e9
            max_next_q = masked_next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Calculate loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
    
    def save_model(self, path):
        torch.save(self.main_net.state_dict(), path)
    
    def load_model(self, path):
        self.main_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.main_net.state_dict())