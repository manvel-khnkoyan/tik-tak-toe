# play.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.environment import TicTacToeEnv
from src.strategy.qlearn.qlearn_agent import QLearAgent
import pickle
import random

# Load the trained Q-table
with open("src/strategy/qlearn/__models__/model-target.pkl", "rb") as f:
#with open("src/strategy/qlearn/__models__/model-20241130.pkl", "rb") as f:
    q_table = pickle.load(f)

# Initialize environment and agent
env = TicTacToeEnv()
agent = QLearAgent(alpha=0, gamma=0, epsilon=0, q_table=q_table)

state = env.reset()
done = False
current_player = random.choice([1, -1])

while not done:
    # Display the current board
    env.render()
    print("\n")

    board_input = state.copy()
    if current_player == -1:
        board_input = -board_input  # Reverse the board for the agent

    if current_player == 1:
        # Agent's turn
        print("Agent's turn:")
        action = agent.choose_action(board_input)
    else:
        # User's turn
        print("Your turn:")
        available_actions = agent.get_available_actions(board_input)
        action = None
        while action not in available_actions:
            try:
                user_input = input("Enter your move (1-9, where 1 is top-left and 9 is bottom-right): ")
                user_move = int(user_input) - 1  # Convert to 0-based index
                row = user_move // 3
                col = user_move % 3
                action = (row, col)
                if action not in available_actions:
                    print("Cell is already occupied or invalid. Please choose another cell.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a number from 1 to 9.")
                action = None

    # Perform the action in the environment
    state = env.step(action, current_player)
    winner = env.check_winner()

    done = winner is not None
    current_player *= -1  # Switch player

# Display the final board and the result
env.render()
print("\n")

if winner == 1:
    print("Agent wins! ❌")
elif winner == -1:
    print("You win! ✅")
else:
    print("It's a draw! 🤝")