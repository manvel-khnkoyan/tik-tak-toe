import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from agent_base import BaseAgent

class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()
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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent(BaseAgent):
    def __init__(self, 
            input_size=9, 
            hidden_size=256, 
            output_size=9,
            alpha=0.001,
            gamma=0.99,
            epsilon=1.0,
            batch_size=64,
            buffer_size=5000,
            target_update=10):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        self.main_net = DQNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=alpha, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)

    def learn(self, prev_state, action, reward, next_state):
        # Store experience in canonical form
        canon_prev = self.get_canonical_form(prev_state)
        canon_next = self.get_canonical_form(next_state)
        canon_action = self.transform_action(prev_state, action)
        
        self.buffer.add(
            canon_prev,
            canon_action[0]*3 + canon_action[1],
            reward,
            canon_next
        )

        if len(self.buffer) < self.batch_size:
            return

        # Sample and train
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        # Current Q-values
        current_q = self.main_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            
            # Mask invalid actions
            valid_moves_mask = (next_states == 0).float()
            masked_q = next_q_values * valid_moves_mask
            masked_q[masked_q == 0] = float('-inf')
            
            next_q = masked_q.max(1)[0]
            
            # Detect terminal states
            terminal_mask = torch.tensor(
                [self._is_terminal(s.numpy()) for s in next_states],
                dtype=torch.float32
            ).to(self.device)
            
            target_q = rewards + (1 - terminal_mask) * self.gamma * next_q

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def choose_action(self, board):
        canonical_state, _, inverse_transform = self.get_canonical_info(board)
        canonical_board = np.array(canonical_state).reshape(3, 3)
        available_actions = self.get_available_actions(canonical_board)
        
        if not available_actions:
            return None

        if random.random() < self.epsilon:
            canonical_action = random.choice(available_actions)
            return inverse_transform(*canonical_action)

        state_tensor = torch.FloatTensor(canonical_state).to(self.device)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)

        action_indices = [r*3 + c for (r,c) in available_actions]
        best_idx = torch.argmax(q_values[action_indices]).item()
        return inverse_transform(*available_actions[best_idx])

    def save_model(self, path):
        torch.save(self.main_net.state_dict(), path)
    
    def load_model(self, path):
        self.main_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.main_net.state_dict())