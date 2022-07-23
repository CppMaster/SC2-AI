import os
import pickle as pkl
import random
import sys
import time
from pprint import pprint

import optuna
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from minigames.collect_minerals_and_gas.src.command_center_reward_wrapper import CommandCenterRewardWrapper
from minigames.collect_minerals_and_gas.src.env_dicrete import CollectMineralAndGasDiscreteEnv
from minigames.collect_minerals_and_gas.src.refinery_reward_wrapper import RefineryRewardWrapper
from minigames.collect_minerals_and_gas.src.supply_depot_reward_wrapper import SupplyDepotRewardWrapper
from minigames.collect_minerals_and_gas.src.supply_taken_reward_wrapper import SupplyTakenRewardWrapper
from minigames.collect_minerals_and_gas.src.workers_active_reward_wrapper import WorkersActiveRewardWrapper
from optuna_utils.maskable_trial_eval_callback import MaskableTrialEvalCallback
from optuna_utils.sample_params.ppo import sample_ppo_params
from optuna_utils.trial_eval_callback import TrialEvalCallback
from wrappers.reward_scale_wrapper import RewardScaleWrapper

FLAGS = flags.FLAGS
FLAGS(sys.argv)


study_path = "minigames/collect_minerals_and_gas/results/optuna/2"


def objective(trial: optuna.Trial) -> float:

    time.sleep(random.random() * 16)

    step_mul = trial.suggest_categorical("step_mul", [2, 4, 8, 16, 32, 64])
    env_kwargs = {"step_mul": step_mul}
    reward_scale = trial.suggest_loguniform("reward_scale", 0.0001, 1.0)
    worker_active_reward_scale = trial.suggest_uniform("worker_active_reward_scale", 0., 2.)
    supply_taken_reward_scale = trial.suggest_uniform("supply_taken_reward_scale", 0., 2.)
    supply_depot_reward_scale = trial.suggest_uniform("supply_depot_reward_scale", 0., 2.)
    supply_free_margin = trial.suggest_int("supply_free_margin", 0, 8)
    cc_reward_scale = trial.suggest_uniform("cc_reward_scale", 0., 2.)
    cc_time_margin = trial.suggest_uniform("cc_time_margin", 0., 300.)
    refinery_reward_scale = trial.suggest_uniform("refinery_reward_scale", 0., 2.)
    refinery_worker_slots_margin = trial.suggest_int("refinery_worker_slots_margin", 0, 8)
    refinery_suboptimal_worker_slot_weight = trial.suggest_uniform("refinery_suboptimal_worker_slot_weight", 0., 1.)
    reward_params = {
        "reward_scale": reward_scale,
        "worker_active_reward_scale": worker_active_reward_scale,
        "supply_taken_reward_scale": supply_taken_reward_scale,
        "supply_depot_reward_scale": supply_depot_reward_scale,
        "supply_free_margin": supply_free_margin,
        "cc_reward_scale": cc_reward_scale,
        "cc_time_margin": cc_time_margin,
        "refinery_reward_scale": refinery_reward_scale,
        "refinery_worker_slots_margin": refinery_worker_slots_margin,
        "refinery_suboptimal_worker_slot_weight": refinery_suboptimal_worker_slot_weight
    }

    sampled_hyperparams = sample_ppo_params(trial)

    path = f"{study_path}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    env = CollectMineralAndGasDiscreteEnv(**env_kwargs)
    env = Monitor(env)
    env = WorkersActiveRewardWrapper(
        env,
        mineral_reward=100.*worker_active_reward_scale,
        lesser_mineral_reward=50.*worker_active_reward_scale,
        gas_reward=75.*worker_active_reward_scale
    )
    env = SupplyTakenRewardWrapper(env, reward_diff=100.*supply_taken_reward_scale)
    env = SupplyDepotRewardWrapper(
        env, reward_diff=100.*supply_depot_reward_scale, free_supply_margin=supply_free_margin
    )
    env = CommandCenterRewardWrapper(env, reward_diff=10.*cc_reward_scale, time_margin=cc_time_margin)
    env = RefineryRewardWrapper(
        env,
        reward_diff=100.*refinery_reward_scale,
        workers_slots_margin=refinery_worker_slots_margin,
        suboptimal_worker_slot_weight=refinery_suboptimal_worker_slot_weight
    )
    env = RewardScaleWrapper(env, reward_scale)
    model = MaskablePPO("MlpPolicy", env=env, seed=None, verbose=0, tensorboard_log=path, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=3, verbose=1)
    eval_callback = MaskableTrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=10, eval_freq=50000, deterministic=False, callback_after_eval=stop_callback
    )

    params = env_kwargs | sampled_hyperparams | reward_params
    with open(f"{path}/params.json", "w") as f:
        f.write(str(params))

    try:
        model.learn(10000000, callback=eval_callback)
        env.close()
    except (AssertionError, ValueError) as e:
        env.close()
        print(e)
        print("============")
        print("Sampled params:")
        pprint(params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward

    model.save(f"{path}/last_model.zip")

    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


if __name__ == "__main__":

    sampler = TPESampler(n_startup_trials=5, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(objective, n_jobs=4, n_trials=128)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print("Value: ", best_trial.value)

    print("Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    with open(f"{study_path}/study.pkl", "wb+") as f:
        pkl.dump(study, f)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)
