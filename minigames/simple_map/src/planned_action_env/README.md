# Planned-Action Simple Map

Custom PySC2 planned-action environment for the SimpleMap mini-game, along with reward shapers and training scripts.

## Key scripts
- `train.py` — main training loop for the showcased runs.
- `eval.py` — evaluation helper.
- `env.py` — core environment definition.
- `reward_shaper.py`, `score_reward_wrapper.py`, `worker_reward_shaper.py`, `supply_depot_reward_shaper.py` — reward shaping utilities.
- `combined_extractor.py`, `spatial_extractor.py` — feature extractors.

## Run training
```bash
python minigames/simple_map/src/planned_action_env/train.py
```

## Notes
- Requires StarCraft II assets and maps available to PySC2.
- Logs and checkpoints are written under this mini-game directory; clear them between experiments if needed.

