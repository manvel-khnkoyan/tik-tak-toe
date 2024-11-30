# evaluate_agents.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.environment import TicTacToeEnv
from src.strategy.qlearn.qlearn_agent import QLearAgent
import pickle

# Number of evaluation games
n = 10000  # You can adjust this number as needed

# Load the trained Q-tables
with open("src/strategy/qlearn/__models__/model-20241130.pkl", "rb") as f:
    q_table1 = pickle.load(f)
with open("src/strategy/qlearn/__models__/model-target.pkl", "rb") as f:
    q_table2 = pickle.load(f)

# Initialize agents with epsilon=0 to disable exploration
agent1 = QLearAgent(alpha=0, gamma=0, epsilon=0, q_table=q_table1)
agent2 = QLearAgent(alpha=0, gamma=0, epsilon=0, q_table=q_table2)

# Initialize the environment
env = TicTacToeEnv()

# Performance tracking
wins_agent1 = 0
wins_agent2 = 0
draws = 0

for game in range(n):
    state = env.reset()
    player = 1 if game % 2 == 0 else -1
    
    while True:
        if player == 1:
            # Agent1's turn
            action = agent1.choose_action(state)
        else:
            # Agent2's turn
            action = agent2.choose_action(-state)
        
        # Perform the action
        next_state = env.step(action, player)
        winner = env.check_winner()
        
        if winner is not None:
            if winner == 0:
                draws += 1
            elif winner == 1:
                wins_agent1 += 1
            else:
                wins_agent2 += 1
            break  # Game over
        else:
            # Game continues
            state = next_state
            player *= -1  # Switch player

# Output the results
print(f"After {n} games:")
print(f"Agent1 wins: {wins_agent1}")
print(f"Agent2 wins: {wins_agent2}")
print(f"Draws: {draws}")