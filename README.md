# Tik-Tac-Toe: Reinforcement Learning

This project implements Deep Q-Learning (DQN) and Q-Learning for training an agent in a given environment.

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

## Training Agents

#### Train using Q-Learning

```bash
python3 src/qlearn/qlearn_train.py
```

#### Train using Deep Q-Learning (DQN)

```bash
python3 src/dqn/dqn_train.py
```

## Playing the Trained Model

After training, you can play using the trained model (adjust the script accordingly if needed):

```bash
python3 src/play.py --player1=human --player2=human
```

options for player1 and player2 are: `human`, `dqn`, `qlearn`
