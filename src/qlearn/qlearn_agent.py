import numpy as np
import random
import pickle
from base_agent import BaseAgent

class QLearAgent(BaseAgent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(alpha, gamma, epsilon)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def choose_action(self, state):
        # Convert state to a hashable tuple representation
        state = tuple(np.array(state).flatten())
        board = np.array(state).reshape(3, 3)
        actions = self.get_available_actions(board)
        if not actions:
            return None

        # With probability epsilon, choose a random action (exploration)
        if random.random() < self.epsilon:
            return random.choice(actions)

        # Initialize Q-values for unseen states
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in actions}

        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, prev_state, action, reward, next_state, done):
        # Convert states to tuple representations
        prev_state = tuple(prev_state.flatten())
        next_state = tuple(next_state.flatten())

        if prev_state not in self.q_table:
            board = np.array(prev_state).reshape(3, 3)
            available_actions = self.get_available_actions(board)
            self.q_table[prev_state] = {a: 0 for a in available_actions}

        if action not in self.q_table[prev_state]:
            raise ValueError(f"Invalid action {action} for state {prev_state}")

        next_board = np.array(next_state).reshape(3, 3)
        next_actions = self.get_available_actions(next_board)
        next_max_q = max([self.q_table.get(next_state, {}).get(a, 0) for a in next_actions]) if next_actions else 0

        current_q = self.q_table[prev_state].get(action, 0)
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[prev_state][action] = new_q

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)