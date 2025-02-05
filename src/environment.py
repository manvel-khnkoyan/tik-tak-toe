# environment.py
import numpy as np

class TictactoeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)

    def reset(self):
        self.board.fill(0)
        return self.board
    
    def state(self):
        return self.board

    def step(self, action, player):
        row, col = action
        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move: Cell ({row}, {col}) is already occupied.")
        self.board[row, col] = player

        return self.board

    def check_winner(self):
        for player in [1, -1]:
            # Check rows and columns
            if any(np.all(self.board[i, :] == player) for i in range(3)) or \
               any(np.all(self.board[:, j] == player) for j in range(3)):
                return player
            # Check diagonals
            if np.all(np.diag(self.board) == player) or \
               np.all(np.diag(np.fliplr(self.board)) == player):
                return player
        # Check for a draw
        if not np.any(self.board == 0):
            return 0  # Game is a draw
        return None  # Game is ongoing
    
    def get_available_actions(self):
        # Assuming self.board is a list or array of size 9, with 0 indicating empty
        available = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    available.append((i, j))
        return available

    def render(self):
        #symbols = {1: '❌', -1: '⭕'}
        symbols = {1: ' X', -1: ' O'}
        counter = 1  # Start block numbering from 1
        for row in self.board:
            items = []
            for cell in row:
                items.append(symbols.get(cell,  str(counter) + ' '))
                counter += 1
            print(' | '.join(items))
            print('-' * 9)