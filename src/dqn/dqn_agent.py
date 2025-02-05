# Import necessary libraries
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from base_agent import BaseAgent

# Define Deep Q-Network architecture
class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

# Experience replay buffer with 'done' flag support
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque()
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise IndexError("Batch size exceeds buffer size")
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

# Enhanced DQN Agent that assumes states are already canonical.
# The BaseAgent’s canonical methods will be used externally.
class DQNAgent(BaseAgent):
    def __init__(self, 
                 input_size=9,
                 hidden_size=256,
                 output_size=9,
                 alpha=0.001,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 target_update=64):
        
        # Initialize BaseAgent (which holds canonical_transform logic)
        super().__init__(alpha, gamma, epsilon)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Initialize networks
        self.main_net   = DQNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQNetwork(input_size, hidden_size, output_size).to(self.device)

        self.optimizer  = optim.Adam(self.main_net.parameters(), lr=alpha, weight_decay=0)
        self.loss_fn    = nn.SmoothL1Loss()
        self.buffer     = ReplayBuffer()

    def learn(self, prev_state, action, reward, next_state, done):
        """
        Expects:
          - prev_state and next_state: canonical representations (e.g. a 9-element array or equivalent)
          - action: a canonical (row, col) tuple
        """
        # Flatten the canonical states for the network
        prev_state_flat = np.array(prev_state).flatten()
        next_state_flat = np.array(next_state).flatten()
        
        # Convert canonical action (row, col) to a single index
        action_idx = action[0] * 3 + action[1]
        
        # Update the learning rate // self.alpha is being updated by the trainer script
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.alpha
        
        # Store the experience
        self.buffer.add(prev_state_flat, action_idx, reward, next_state_flat, done)

        if len(self.buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calculate Q-values for current and next states
        current_q = self.main_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q    = self.target_net(next_states).max(1)[0].detach()
        target_q  = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and perform backpropagation
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Logging ↓↓↓
        # Clip gradients to avoid explosion
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.main_net.parameters(), 
            0.5
        ).item()
        if self.update_counter % 100 == 0:
            print(f"Gradient Norm: {total_grad_norm:.4f}")
            print(f"Current Q: {current_q.mean().item():.4f} | Target Q: {target_q.mean().item():.4f}")
            print(f"Loss: {loss.item():.4f}")
        # Logging ↑↑↑

        # Update the main network
        self.optimizer.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def choose_action(self, canonical_state):
        """
        Expects:
          - canonical_state: already transformed to canonical form.
          Returns a canonical (row, col) action.
        """
        canonical_flat = np.array(canonical_state).flatten()
        canonical_board = np.array(canonical_state).reshape(3, 3)
        if not np.any(canonical_board == 0):
            return None
        
        available_actions = self.get_available_actions(canonical_board)
        if not available_actions:
            return None

        # With probability epsilon, choose a random valid action
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Otherwise, choose the action with the highest Q-value among valid ones
        state_tensor = torch.FloatTensor(canonical_flat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_net(state_tensor).squeeze(0).cpu().numpy()
            # Map available (row, col) actions to indices
            action_indices = [r * 3 + c for (r, c) in available_actions]
            best_index = np.argmax(q_values[action_indices])
            return available_actions[best_index]

    def save_model(self, path):
        torch.save({
            'main_net': self.main_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.main_net.load_state_dict(checkpoint['main_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])