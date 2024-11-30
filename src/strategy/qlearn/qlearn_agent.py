import numpy as np
import random
from src.base_agent import BaseAgent

class QLearAgent(BaseAgent):
    def __init__(self, alpha, gamma, epsilon, q_table=None):
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.q_table = q_table if q_table is not None else {}  # Initialize Q-table

    def get_actions(self, board):
        return tuple(board.flatten())

    def get_available_actions(self, board):
        return [tuple(map(int, pos)) for pos in zip(*np.nonzero(board == 0))]

    def choose_action(self, board):
        state_key = self.get_actions(board)
        available_actions = self.get_available_actions(board)

        if not available_actions:
            return None  # No possible actions (board is full)

        # Exploration: choose a random available action
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Exploitation: choose the best known action
        q_values = self.q_table.get(state_key)
        if q_values is None:
            q_values = {tuple(map(int, a)): 0 for a in available_actions}
            self.q_table[state_key] = q_values

        # Get Max Q-value and select corresponding action(s)
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]

        # Choose randomly among the best actions
        return random.choice(best_actions)

    def learn(self, prev_state_board, action, reward, next_state_board):
        # Convert boards to state keys
        state_key = self.get_actions(prev_state_board)
        next_state_key = self.get_actions(next_state_board)
        
        # Get available actions for current and next states
        available_actions = self.get_available_actions(prev_state_board)
        next_available_actions = self.get_available_actions(next_state_board)
        
        # Initialize Q-values for the current state if not already present
        if state_key not in self.q_table:
            self.q_table[state_key] = {tuple(map(int, a)): 0 for a in available_actions}
        
        # Initialize Q-values for the next state if not already present and there are available actions
        if next_state_key not in self.q_table and next_available_actions:
            self.q_table[next_state_key] = {tuple(map(int, a)): 0 for a in next_available_actions}
        
        # Normalize action to a tuple of ints
        action = tuple(map(int, action))
        
        # Current Q-value for the taken action
        current_q = self.q_table[state_key][action]
        
        # Maximum Q-value for the next state
        next_max_q = 0
        if next_available_actions and next_state_key in self.q_table:
            next_max_q = max(self.q_table[next_state_key].values())
        else:
            next_max_q = 0  # Terminal state (no future actions)
        
        # Update the Q-value using the Q-learning formula
        self.q_table[state_key][action] = current_q + self.alpha * (
            reward + self.gamma * next_max_q - current_q
        )