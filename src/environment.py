# environment.py
import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)

    def reset(self):
        """Resets the game board to the initial state."""
        self.board.fill(0)
        return self.board
    
    def state(self):
        return self.board

    def step(self, action, player):
        """
        Updates the game board with the player's action.

        Parameters:
        - action: A tuple (row, col) indicating the cell to mark.
        - player: The player's symbol (1 for 'X' and -1 for 'O').

        Returns:
        - next_state: The updated game board.
        - winner: The winner of the game (1, -1, 0 for draw, or None if the game is ongoing).
        """
        row, col = action
        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move: Cell ({row}, {col}) is already occupied.")
        self.board[row, col] = player

        return self.board

    def check_winner(self):
        """Checks the current board for a winner or a draw."""
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

    def render(self):
        symbols = {1: '❌', -1: '⭕'}
        counter = 1  # Start block numbering from 1
        for row in self.board:
            items = []
            for cell in row:
                items.append(symbols.get(cell,  str(counter) + ' '))
                counter += 1
            print(' | '.join(items))
            print('-' * 9)