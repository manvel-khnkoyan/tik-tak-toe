import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from environment import TictactoeEnv
from qlearn_agent import QLearAgent

def linear_decay(start, end, step, total):
    return start - (start - end) * (step / total)

def exponential_decay(start, end, step, total):
    return start * (end / start) ** (step / total)

if __name__ == "__main__":

    # Initialize environment
    env = TictactoeEnv()

    # Training parameters
    num_episodes = 5000
    num_episodes_batch = 100

    # Hyperparameters with decay
    alpha_start, alpha_end = 0.9, 0.1
    gamma_start, gamma_end = 0.7, 0.9
    epsilon_start, epsilon_end = 1.0, 0.1

    agent1 = QLearAgent(alpha=alpha_start, gamma=gamma_start, epsilon=epsilon_start)
    agent2 = QLearAgent(alpha=0, gamma=0, epsilon=0)

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
                last_action = agent1.choose_action_canonical(env.board)
                
                env.step(last_action, current_player)
                winner = env.check_winner()
                
                if winner is not None:
                    done = True

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
                    
                    agent1.learn_canonical(prev_state, last_action, reward, next_state, done)

                else:
                    # Switch to opponent's turn
                    # It will be learnt by opponent move
                    current_player = -1

            else:
                # Agent2's turn (random moves)
                available_actions = env.get_available_actions()
                if available_actions:  # Ensure there are valid moves
                    action = random.choice(available_actions)
                    env.step(action, current_player)
                winner = env.check_winner()
                
                if winner is not None:
                    done = True

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
                        agent1.learn_canonical(prev_state, last_action, reward, next_state, done)
                    
                else:
                    # Game continues, learn from transition
                    if prev_state is not None:
                        next_state = np.copy(env.board)
                        agent1.learn_canonical(prev_state, last_action, 0, next_state, done)

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

    # Save the model
    model_dir = "src/__models__"
    os.makedirs(model_dir, exist_ok=True)
    agent1.save_model(os.path.join(model_dir, "qlearn.pkl"))
    print("Training completed!")