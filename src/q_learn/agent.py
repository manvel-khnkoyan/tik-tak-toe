import numpy as np
import random
import pickle
from src.base_agent import BaseAgent

class QLearAgent(BaseAgent):
    def __init__(self, alpha, gamma, epsilon, q_table=None):
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.q_table = q_table if q_table is not None else {}

    def get_canonical_form(self, board):
        board = board.reshape(3, 3)
        symmetries = [
            board, np.rot90(board), np.rot90(board, 2), np.rot90(board, 3),
            np.fliplr(board), np.fliplr(np.rot90(board)), 
            np.fliplr(np.rot90(board, 2)), np.fliplr(np.rot90(board, 3))
        ]
        canonical_candidates = []
        for sym in symmetries:
            flat_sym = sym.astype(int).flatten()
            tuple_sym = tuple(int(x) for x in flat_sym)
            canonical_candidates.append(tuple_sym)
        return min(canonical_candidates)

    def get_canonical_info(self, board):
        board = board.reshape(3, 3)
        transforms = [
            (lambda r, c: (r, c), lambda r, c: (r, c)),          # Identity
            (lambda r, c: (c, 2 - r), lambda r, c: (2 - c, r)),  # Rot90
            (lambda r, c: (2 - r, 2 - c), lambda r, c: (2 - r, 2 - c)),  # Rot180
            (lambda r, c: (2 - c, r), lambda r, c: (c, 2 - r)),  # Rot270
            (lambda r, c: (r, 2 - c), lambda r, c: (r, 2 - c)),  # Flip L-R
            (lambda r, c: (2 - r, c), lambda r, c: (2 - r, c)),  # Flip U-D
            (lambda r, c: (c, r), lambda r, c: (c, r)),          # Transpose
            (lambda r, c: (2 - c, 2 - r), lambda r, c: (2 - c, 2 - r)),  # Anti-Transpose
        ]

        min_state = None
        best_transform = None
        best_inverse = None
        for t, inv in transforms:
            transformed = np.zeros_like(board)
            for r in range(3):
                for c in range(3):
                    tr, tc = t(r, c)
                    transformed[tr][tc] = board[r][c]
            state_tuple = tuple(transformed.flatten().astype(int))
            if (min_state is None) or (state_tuple < min_state):
                min_state = state_tuple
                best_transform = t
                best_inverse = inv
        return min_state, best_transform, best_inverse

    def get_available_actions(self, board):
        return [tuple(int(x) for x in pos) for pos in zip(*np.nonzero(board == 0))]

    def choose_action(self, board):
        # Get canonical form and inverse transformation
        canonical_state, _, inverse_transform = self.get_canonical_info(board)
        canonical_board = np.array(canonical_state).reshape(3, 3)
        canonical_actions = self.get_available_actions(canonical_board)

        if not canonical_actions:
            return None

        # Exploration: Random canonical action -> original coordinates
        if random.random() < self.epsilon:
            canonical_action = random.choice(canonical_actions)
            original_action = inverse_transform(*canonical_action)
            return original_action

        # Exploitation: Use Q-table with canonical state
        if canonical_state not in self.q_table:
            self.q_table[canonical_state] = {a: 0 for a in canonical_actions}

        q_values = self.q_table[canonical_state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        canonical_action = random.choice(best_actions)
        original_action = inverse_transform(*canonical_action)
        return original_action

    def learn(self, prev_state_board, action, reward, next_state_board):
        # Convert previous state to canonical form
        prev_canonical, transform, _ = self.get_canonical_info(prev_state_board)
        
        # Convert action to canonical coordinates
        action_row, action_col = action
        canonical_action = transform(action_row, action_col)
        
        # Get next state's canonical form
        next_canonical = self.get_canonical_form(next_state_board)

        # Initialize Q-values if needed
        if prev_canonical not in self.q_table:
            canonical_board = np.array(prev_canonical).reshape(3, 3)
            available_actions = self.get_available_actions(canonical_board)
            self.q_table[prev_canonical] = {a: 0 for a in available_actions}

        # Validate action
        if canonical_action not in self.q_table[prev_canonical]:
            raise ValueError(f"Invalid action {canonical_action} for state {prev_canonical}")

        # Calculate next_max_q
        next_canonical_board = np.array(next_canonical).reshape(3, 3)
        next_actions = self.get_available_actions(next_canonical_board)
        next_max_q = max([self.q_table.get(next_canonical, {}).get(a, 0) for a in next_actions]) if next_actions else 0

        # Q-learning update
        current_q = self.q_table[prev_canonical].get(canonical_action, 0)
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[prev_canonical][canonical_action] = new_q

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)