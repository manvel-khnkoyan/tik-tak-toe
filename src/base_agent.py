import numpy as np

class BaseAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        # This is a base class constructor. Derived classes should implement their own initialization.
        pass

    def learn(self, prev_state, action, reward, next_state, done):
        raise NotImplementedError

    def choose_action(self, board):
        raise NotImplementedError
    
    def load_model(self, path):
        raise NotImplementedError
    
    def save_model(self, path):
        raise NotImplementedError

    def get_available_actions(self, board):
        return [tuple(pos) for pos in zip(*np.nonzero(board == 0))]

    def learn_canonical(self, prev_state_board, action, reward, next_state_board, done):
        prev_state, transform, _ = self.canonical_transform(prev_state_board)
        canonical_action = transform(*action)
        next_state, _, _ = self.canonical_transform(next_state_board)
        self.learn(prev_state, canonical_action, reward, next_state, done)
    
    def choose_action_canonical(self, board):
        state, _, inverse_transform = self.canonical_transform(board)
        canonical_action = self.choose_action(state)
        if canonical_action is None:
            return None

        return inverse_transform(*canonical_action)

    def canonical_transform(self, board):
        board = board.reshape(3, 3)
        transforms = [
            (lambda r, c: (r, c),           lambda r, c: (r, c)),           # Identity
            (lambda r, c: (c, 2 - r),       lambda r, c: (2 - c, r)),       # Rot90
            (lambda r, c: (2 - r, 2 - c),   lambda r, c: (2 - r, 2 - c)),   # Rot180
            (lambda r, c: (2 - c, r),       lambda r, c: (c, 2 - r)),       # Rot270
            (lambda r, c: (r, 2 - c),       lambda r, c: (r, 2 - c)),       # Flip L-R
            (lambda r, c: (2 - r, c),       lambda r, c: (2 - r, c)),       # Flip U-D
            (lambda r, c: (c, r),           lambda r, c: (c, r)),           # Transpose
            (lambda r, c: (2 - c, 2 - r),   lambda r, c: (2 - c, 2 - r)),   # Anti-Transpose
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

        return np.array(min_state).reshape(3, 3), best_transform, best_inverse