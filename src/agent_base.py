import numpy as np

"""
Each agent play as player N: 1
So any time agent is making a move, you need use correct env
- if needed you may need to reverse the env.board
"""
class BaseAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        # This is a base class, so the __init__ method is intentionally left empty.
        pass

    def learn(self, prev_state_board, action, reward, next_state_board):
        raise NotImplementedError

    def choose_action(self, env):
        raise NotImplementedError
    
    def load_model(self, path):
        raise NotImplementedError
    
    def save_model(self, path):
        raise NotImplementedError
    
    def get_available_actions(self, board):
        return [tuple(pos) for pos in zip(*np.nonzero(board == 0))]
    
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

        return min_state, best_transform, best_inverse
