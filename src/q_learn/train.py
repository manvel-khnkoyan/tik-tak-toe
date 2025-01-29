# train.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.environment import TictactoeEnv
from src.q_learn.agent import QLearAgent
import random
import numpy as np

# Initialize environment
env = TictactoeEnv()

# Training parameters
num_episodes = 10000
num_episodes_batch = 100

# Hyperparameters with decay
alpha_start, alpha_end = 0.9, 0.1
gamma_start, gamma_end = 0.7, 0.9
epsilon_start, epsilon_end = 1.0, 0.01

agent1 = QLearAgent(alpha=alpha_start, gamma=gamma_start, epsilon=epsilon_start)
agent2 = QLearAgent(alpha=0, gamma=0, epsilon=0)  # Random agent

def linear_decay(start, end, episode, total_episodes):
    return start - (start - end) * (episode / total_episodes)

def exponential_decay(start, end, episode, total_episodes):
    return start * (end / start) ** (episode / total_episodes)

# Performance tracking
total_wins = 0
total_loss = 0
total_draw = 0

for episode in range(num_episodes):
    # Decay parameters
    agent1.alpha = linear_decay(alpha_start, alpha_end, episode, num_episodes)
    agent1.gamma = exponential_decay(gamma_start, gamma_end, episode, num_episodes)
    agent1.epsilon = exponential_decay(epsilon_start, epsilon_end, episode, num_episodes)
    
    # Force exploitation in final phase
    if num_episodes - episode <= num_episodes_batch:
        agent1.epsilon = 0

    state = env.reset()
    current_player = random.choice([1, -1])
    
    prev_state = None
    last_action = None
    
    done = False
    action_num = 0

    while not done:
        action_num += 1
        
        if current_player == 1:
            # Agent1's turn
            prev_state = np.copy(env.board)
            last_action = agent1.choose_action(env.board)
            
            env.step(last_action, current_player)
            winner = env.check_winner()
            
            if winner is not None:
                # Game ended by Agent1's move
                if winner == 1:
                    reward = 1.0
                    total_wins += 1
                elif winner == -1:
                    reward = -1.0
                    total_loss += 1
                else:
                    reward = 0.5
                    total_draw += 1
                next_state = np.copy(env.board)
                agent1.learn(prev_state, last_action, reward, next_state)
                done = True
            else:
                # Switch to opponent's turn
                current_player = -1

        else:
            # Agent2's turn (random moves)
            available_actions = env.get_available_actions()
            if available_actions:  # Ensure there are valid moves
                action = random.choice(available_actions)
                env.step(action, current_player)
            winner = env.check_winner()
            
            if winner is not None:
                # Game ended by Agent2's move
                if winner == 1:
                    reward = 1.0
                    total_wins += 1
                elif winner == -1:
                    reward = -1.0
                    total_loss += 1
                else:
                    reward = 0.5
                    total_draw += 1
                if prev_state is not None:
                    next_state = np.copy(env.board)
                    agent1.learn(prev_state, last_action, reward, next_state)
                done = True
            else:
                # Game continues, learn from transition
                if prev_state is not None:
                    next_state = np.copy(env.board)
                    agent1.learn(prev_state, last_action, 0, next_state)
                # Switch back to Agent1's turn
                current_player = 1

    # Progress reporting
    if (episode + 1) % num_episodes_batch == 0:
        win_rate = total_wins / num_episodes_batch * 100
        loss_rate = total_loss / num_episodes_batch * 100
        draw_rate = total_draw / num_episodes_batch * 100
        
        print(f"Episode {episode+1}/{num_episodes}")
        print(f"Win rate: {win_rate:.1f}% | Loss rate: {loss_rate:.1f}% | Draw rate: {draw_rate:.1f}%")
        print(f"Alpha: {agent1.alpha:.3f} | Gamma: {agent1.gamma:.3f} | Epsilon: {agent1.epsilon:.3f}")
        print("----------------------------------")
        
        # Reset counters
        total_wins = total_loss = total_draw = 0

# Save trained model
agent1.save_model("src/qlearn-model.pkl")
print("Training completed!")