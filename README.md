Cat Eats Fish – DQN Reinforcement Learning

This project was developed for CDS524 – Machine Learning for Business.

Project Overview

I implemented a Deep Q-Network (DQN) agent to solve a custom 2D game environment where a cat collects fish while avoiding traps.

Environment Design
Action Space
5 discrete actions:
- Up
- Down
- Left
- Right
- Stay

State Space (11 dimensions)
- Cat position (normalized)
- Cat velocity
- Relative position to fish
- Relative position to nearest trap
- Normalized time step

Reward Function
- Small survival reward
- Distance-based shaping reward
- +3 for eating fish
- -8 for hitting trap

Model Architecture

Deep Q-Network:
- 128 neurons
- 128 neurons
- 64 neurons
- Output layer (5 Q-values)

Hyperparameters

- Learning rate: 0.001
- Discount factor: 0.95
- Epsilon-greedy exploration

How to Run

```bash
python3 main.py --mode play
