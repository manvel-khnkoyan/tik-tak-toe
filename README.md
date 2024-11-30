# Tic-Tac-Toe with Reinforcement Learning

## Project Description
This project is a Python-based implementation of the classic Tic-Tac-Toe game, enhanced with a reinforcement learning agent using Q-learning. The agent learns optimal strategies by playing games and refining its decision-making based on the outcomes.

The project includes:
1. A **Tic-Tac-Toe environment** simulating the game's mechanics.
2. A **Q-learning agent** that can train itself and compete against human players or other agents.

---

## Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/manvel-khnkoyan/tic-tac-toe.git
cd tic-tac-toe
```

2. Install the required packages:

```bash
python3 -m venv env
```

3. Activate the virtual environment:
```bash
source env/bin/activate
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

5. Play the game:
```bash
python3 src/simulations/human_vs_qagent.py
```

## Features
- 🎮 **Interactive Gameplay**: Play against the Q-learning agent and test its strategies.
- 🤖 **Reinforcement Learning Integration**: The agent improves its strategy over time using Q-learning.
- ⚙️ **Customizable Training**: Adjust hyperparameters like:
  - **Learning Rate (α):** Controls how much the agent learns from new experiences.
  - **Discount Factor (γ):** Determines the importance of future rewards.
  - **Exploration Rate (ε):** Balances exploration and exploitation during training.
