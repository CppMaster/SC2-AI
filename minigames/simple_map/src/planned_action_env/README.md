# Planned-Action Simple Map

Custom PySC2 planned-action environment for the SimpleMap mini-game, along with reward shapers and training scripts.

## Key scripts
- `train.py` — main training loop for the showcased runs.
- `eval.py` — evaluation helper.
- `env.py` — core environment definition.
- `reward_shaper.py`, `score_reward_wrapper.py`, `worker_reward_shaper.py`, `supply_depot_reward_shaper.py` — reward shaping utilities.
- `combined_extractor.py`, `spatial_extractor.py` — feature extractors.

## How PlannedActionEnv works
- Discrete macro actions (see `ActionIndex` in `env.py`) cover production, tech, upgrades, and army commands like attack, gather, stop, or retreat.
- Each action is validated for prerequisites (minerals, gas, required buildings/tech) and queued if needed; invalid actions are masked with requirements signals in the observation.
- Observations combine scalar economy/army state (e.g., minerals, supply, units, upgrades, enemy proximity) with optional spatial features from the combined/spatial extractors.
- A difficulty scheduler can scale enemy strength over time, and reward shaping modules add dense feedback for economy, supply, upgrades, and combat goals.
- The environment issues the low-level PySC2 commands for the chosen macro action, letting the agent learn high-level planning rather than micro.

## Run training
```bash
python minigames/simple_map/src/planned_action_env/train.py
```

## Notes
- Requires StarCraft II assets and maps available to PySC2.
- Logs and checkpoints are written under this mini-game directory; clear them between experiments if needed.

