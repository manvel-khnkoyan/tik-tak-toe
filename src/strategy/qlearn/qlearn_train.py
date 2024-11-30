# train.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.environment import TicTacToeEnv
from src.strategy.qlearn.qlearn_agent import QLearAgent

import pickle
import random
import argparse
from datetime import datetime

# Argument Parsing
parser = argparse.ArgumentParser(description="Train two Q-learning agents for Tic Tac Toe.")
parser.add_argument("--model-name", type=str, help="Base name of the model files. Timestamp will be appended.")
args = parser.parse_args()

# Generate model name with timestamp
timestamp = datetime.now().strftime("%Y%m%d")  # Format: %Y%m%d_%H%M%S
model_name = args.model_name or "model"  # Default to "model" if no name provided

# Define the directory to save models
model_dir = os.path.join("src", "strategy", "qlearn", "__models__")
os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

# Initialize environment
env = TicTacToeEnv()

# Training parameters
num_episodes = 100000
num_episodes_batch = 100

# Learning parameters

# Alpha controls how much the agent learns from new experiences. 
# It’s a weight given to the most recent feedback compared to the existing knowledge in the Q-table.
alpha_range = (0.9, 1)

# Gamma determines how much importance the agent places on future rewards compared to immediate rewards. 
# It controls the balance between short-term and long-term gains.
gamma_range = (0.7, 0.9)

# Epsilon determines the balance between exploration (trying new actions to discover better strategies) and 
# exploitation (choosing actions based on existing knowledge in the Q-table).
epsilon_range = (1, 0.5)

def linear_decay(start, end, decay):
    return start - (start - end) * decay

def exponential_decay(start, end, decay):
    return start * (end / start) ** decay

def reward_decay(action_num):
    return 9 / action_num

# Load the trained Q-table
q_table = {}
with open("src/strategy/qlearn/__models__/model-target.pkl", "rb") as f:
    q_table = pickle.load(f)

# Initialize agents with their own Q-tables
agentX = QLearAgent(alpha=alpha_range[0], gamma=gamma_range[0], epsilon=epsilon_range[0], q_table={})
agentO = QLearAgent(0, 0, 0, q_table=q_table.copy())

# Performance tracking
total_wins = 0
total_loss = 0
total_draw = 0

# Main training loop
for episode in range(num_episodes):
    decay = episode / num_episodes

    state = env.reset()
    current_player = random.choice([1, -1])

    # Update agent 1's learning parameters
    if current_player == 1:
        agentX.alpha = linear_decay(alpha_range[0], alpha_range[1], decay)
        agentX.gamma = exponential_decay(gamma_range[0], gamma_range[1], decay)
        agentX.epsilon = exponential_decay(epsilon_range[0], epsilon_range[1], decay)

    # Last batch of episodes, set epsilon to 0 to exploate
    if (num_episodes - episode < num_episodes_batch):
        agentX.epsilon = 0


    # Initialize previous state and action
    x_prev_state = env.state().copy()
    x_next_state = x_prev_state
    o_prev_state = x_prev_state

    action_num = 0
    while True:
        action_num += 1

        # Action & State update
        if current_player == 1:
            x_last_action = agentX.choose_action(env.state())
            
            x_prev_state = env.state().copy()
            x_next_state = env.step(x_last_action, current_player).copy()
        else:
            env.step(agentO.choose_action(env.state() * -1), current_player)

        winner = env.check_winner()

        # Game ended
        if winner is not None:
            # Draw
            reward = 0
            reward_amount = 9 / action_num
            if winner == 0:  
                total_draw += 1
                reward = 0.1
            # Current player wins
            elif winner == 1:
                total_wins += 1
                reward = reward_amount
                # Current player loses
            else:
                total_loss += 1
                reward = -reward_amount

            agentX.learn(x_prev_state, x_last_action, reward, x_next_state)

            # Game over
            break
        
        # Game continues
        else:
            if current_player == 1:
                agentX.learn(x_prev_state, x_last_action, 0, x_next_state)

        current_player *= -1

    # Optional: Print progress every num_episodes_batch episodes
    if (episode + 1) % num_episodes_batch == 0:
        rate_wins = round((total_wins / num_episodes_batch) * 100, 2)
        rate_loss = round((total_loss / num_episodes_batch) * 100, 2)
        rate_draw = round((total_draw / num_episodes_batch) * 100, 2)
        print(f"{episode + 1}/{num_episodes}, a({round(agentX.alpha, 2)}), g({round(agentX.gamma, 2)}), e({round(agentX.epsilon, 2)}) Wins: {rate_wins}% ({total_wins}), Losses: {rate_loss}% ({total_loss}), Draws: {rate_draw}% ({total_draw})")

        # Reset performance tracking
        total_wins = 0
        total_loss = 0
        total_draw = 0

# Save the Q-tables to the specified path
with open(os.path.join(model_dir, f"{model_name}-{timestamp}.pkl"), "wb") as f:
    pickle.dump(agentX.q_table, f)

print("Training completed!")