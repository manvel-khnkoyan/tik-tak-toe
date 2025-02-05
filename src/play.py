import random
from environment import TictactoeEnv
from qlearn.qlearn_agent import QLearAgent
from dqn.dqn_agent import DQNAgent

# Agents
#agent1 = 'human'

agent1 = QLearAgent(alpha=0, gamma=0, epsilon=0)
agent1.load_model("__model__qlearn.pkl")

agent2 = DQNAgent(alpha=0, gamma=0, epsilon=0)
agent2.load_model("__model__dqn.pth")

# Environment
env = TictactoeEnv()

# Stats
total_plays = 1000
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
            env.render()
            print("\n")

            available_actions = env.get_available_actions()
            action = None
            while action not in available_actions:
                try:
                    user_input = input("Your move: ")
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
            action = agent.choose_action(board_input)

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