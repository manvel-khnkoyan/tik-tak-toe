import numpy as np
import random

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.strategy.deeplearn.deeplearn_agent import DQNAgent

def check_winner(board):
    """Check if there's a winner on the Tic-Tac-Toe board."""
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    return 0

def is_board_full(board):
    """Check if the board is full (no zeros left)."""
    return not (board == 0).any()

def random_opponent_move(board):
    """Select a random valid move for the opponent."""
    available = list(zip(*np.nonzero(board == 0)))
    return random.choice(available) if available else None

def evaluate_agent(agent, num_episodes=100):
    """Evaluate the agent's performance against a random opponent."""
    wins, losses, draws = 0, 0, 0
    for _ in range(num_episodes):
        board = np.zeros((3, 3), dtype=int)
        done, agent_turn = False, True
        while not done:
            if agent_turn:
                action = agent.choose_action(board)
                if action is None:
                    draws += 1
                    break
                row, col = action
                board[row, col] = 1  # Agent is player 1
                winner = check_winner(board)
                if winner == 1:
                    wins += 1
                    done = True
                elif is_board_full(board):
                    draws += 1
                    done = True
                else:
                    agent_turn = False
            else:
                opp_action = random_opponent_move(board)
                if opp_action is None:
                    draws += 1
                    done = True
                else:
                    row, col = opp_action
                    board[row, col] = -1  # Opponent is -1
                    winner = check_winner(board)
                    if winner == -1:
                        losses += 1
                        done = True
                    elif is_board_full(board):
                        draws += 1
                        done = True
                    else:
                        agent_turn = True
    print(f"Evaluation: Wins={wins}, Losses={losses}, Draws={draws}")
    return wins / num_episodes

def train_agent(episodes=1000, save_path='dqn_model.pth', eval_every=100, eval_episodes=100):
    """Train the DQN agent against a random opponent."""
    agent = DQNAgent(input_size=9, output_size=9)
    
    for episode in range(episodes):
        board = np.zeros((3, 3), dtype=int)
        done = False
        while not done:
            current_state = board.copy()
            action = agent.choose_action(current_state)
            if action is None:  # No move possible (draw)
                break
            
            # Agent's move
            row, col = action
            board[row, col] = 1
            winner = check_winner(board)
            if winner == 1:
                reward, next_state = 1, board.copy()
                done = True
            elif is_board_full(board):
                reward, next_state = 0, board.copy()
                done = True
            else:
                # Opponent's move
                opp_action = random_opponent_move(board)
                if opp_action is None:
                    reward, next_state = 0, board.copy()
                    done = True
                else:
                    row, col = opp_action
                    board[row, col] = -1
                    winner = check_winner(board)
                    if winner == -1:
                        reward, done = -1, True
                    elif is_board_full(board):
                        reward, done = 0, True
                    else:
                        reward, done = 0, False
                    next_state = board.copy()
            
            # Store experience and learn
            agent.store_experience(current_state, action, reward, next_state, done)
            agent.learn()
        
        # Periodic evaluation
        if (episode + 1) % eval_every == 0:
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0  # Disable exploration during evaluation
            win_rate = evaluate_agent(agent, eval_episodes)
            print(f"Episode {episode+1}, Win Rate: {win_rate:.2f}, Epsilon: {original_epsilon:.3f}")
            agent.epsilon = original_epsilon
    
    # Save the trained model
    agent.save_model(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_agent(episodes=10000, eval_every=100, eval_episodes=100)