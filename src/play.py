import random
import argparse
from environment import TictactoeEnv
from qlearn.qlearn_agent import QLearAgent
from dqn.dqn_agent import DQNAgent

# Argument parsing
parser = argparse.ArgumentParser(description="Tic-Tac-Toe with AI and Human Players")
parser.add_argument("--player1", choices=["human", "dqn", "qlearn"], help="Choose player 1 type")
parser.add_argument("--player2", choices=["human", "dqn", "qlearn"], help="Choose player 2 type")
parser.add_argument("--episodes", type=int, default=1000, help="Number of games to play")

args = parser.parse_args()

model_dir = "src/__models__"

# Initialize agents based on arguments
def get_agent(player_type):
    if player_type == "human":
        return "human"
    elif player_type == "dqn":
        agent = DQNAgent(alpha=0, gamma=0, epsilon=0)
        agent.load_model(f"{model_dir}/dqn.pth")
        return agent
    elif player_type == "qlearn":
        agent = QLearAgent(alpha=0, gamma=0, epsilon=0)
        agent.load_model(f"{model_dir}/qlearn.pkl")
        return agent
    else:
        raise ValueError("Invalid argument. For help, run 'python play.py -h'")

agent1 = get_agent(args.player1)
agent2 = get_agent(args.player2)


# Environment
env = TictactoeEnv()

# Stats
total_plays = args.episodes
total_play1_wins = 0
total_play2_wins = 0
total_draws = 0

for _ in range(total_plays):
    state = env.reset()
    current_player = random.choice([1, -1])
    done = False
    game_winner = None

    while not done:
        board_input = state.copy()
        if current_player == -1:
            board_input = -board_input  # Reverse the board for the agent

        agent = agent1 if current_player == 1 else agent2

        if agent == 'human':
            print("\n")
            env.render()
            print("\n")

            available_actions = env.get_available_actions()
            action = None
            while action not in available_actions:
                try:
                    user_input = input("Player [1]: " if current_player == 1 else "Player [2]: ")
                    user_move = int(user_input) - 1  # Convert to 0-based index
                    row = user_move // 3
                    col = user_move % 3
                    action = (row, col)
                    if action not in available_actions:
                        print("Cell is already occupied or invalid. Please choose another cell.")
                except (ValueError, IndexError):
                    print("Invalid input. Please enter a number from 1 to 9.")
                    action = None
        else:
            action = agent.choose_action_canonical(board_input)

        # Perform the action
        state = env.step(action, current_player)
        winner = env.check_winner()
        available_actions = env.get_available_actions()
        done = winner is not None or not available_actions

        if done:
            game_winner = winner
        else:
            current_player *= -1  # Switch player if game continues

    # Update statistics
    if game_winner == 1:
        total_play1_wins += 1
    elif game_winner == -1:
        total_play2_wins += 1
    else:
        total_draws += 1

# Display aggregated results
print("\n--- Final Statistics ---")
print(f"Total Games Played: {total_plays}")
print(f"Agent 1 Wins: {total_play1_wins}")
print(f"Agent 2 Wins: {total_play2_wins}")
print(f"Draws: {total_draws}")