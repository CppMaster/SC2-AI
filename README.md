# SC2-AI

This is a hobby project for learning RL: the agents train from scratch without any human supervisor. 

Reinforcement learning agents and utilities for StarCraft II mini-games using PySC2 and Stable-Baselines3. Each mini-game folder contains custom Gym-style environments, reward wrappers, and training scripts for experimenting with different tasks such as MoveToBeacon, CollectMineralShards, SimpleMap, and more. 

Videos of training runs and results live at https://www.youtube.com/@psai5367.

## Example runs

**Beating Zerg:** [Watch on YouTube](https://youtu.be/g54lGquzb1U?si=Z1Zd6yvhHcxULaB9)

[![Beating Zerg](https://img.youtube.com/vi/g54lGquzb1U/hqdefault.jpg)](https://youtu.be/g54lGquzb1U?si=Z1Zd6yvhHcxULaB9 "Play on YouTube")

## Prerequisites
- Python 3.8+ (tested with SB3 1.6.0 and PySC2 3.0.0)
- StarCraft II installed locally with the official mini-games maps available to PySC2
- GPU drivers compatible with the TensorFlow / PyTorch versions in `requirements.txt`

## Setup
1) Clone the repo and create a virtual environment.
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # on Windows
   # source .venv/bin/activate  # on Linux/macOS
   ```
2) Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3) Ensure the SC2 maps are accessible to PySC2 (typically under `StarCraft II/Maps/`).

## Running an example
- Train PPO on MoveToBeacon:
  ```bash
  python minigames/move_to_beacon/src/train.py
  ```
- Run the simple map planned-action example:
  ```bash
  python minigames/simple_map/src/planned_action_env/train.py
  ```
- Evaluate or tweak scripts by editing the corresponding `sample.py` or `env.py` in each mini-game directory.

## Repository layout
- `minigames/` — environments, reward shapers, and training scripts for each SC2 mini-game.
- `wrappers/` — generic Gym/PySC2 wrappers (actions, rewards, stacking).
- `callbacks/` — training callbacks such as early stopping and logging.
- `optuna_utils/` — Optuna study helpers and sample hyperparameter spaces.
- `utils/` — plotting and value utilities.
- `tests/` — lightweight unit tests (e.g., value stack helpers).

## Notes
- Training scripts assume SC2 assets are installed; PySC2 will prompt if maps are missing.
- TensorBoard logs and model checkpoints are written inside each mini-game directory (see `logs/` and `eval/` paths in the scripts). Consider clearing or versioning these directories between runs.
