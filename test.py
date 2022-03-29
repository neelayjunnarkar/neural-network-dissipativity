import ray
from ray import tune
from envs import CartpoleEnv, InvertedPendulumEnv, LearnedInvertedPendulumEnv, LinearizedInvertedPendulumEnv, PendubotEnv, VehicleLateralEnv, PowergridEnv, OtherInvertedPendulumEnv
from models.RNN import RNNModel
from models.ProjRNN import ProjRNNModel
from models.ProjREN import ProjRENModel
from models.ProjRNNOld import ProjRNNOldModel
from ray.rllib.agents import ppo, pg
import numpy as np
from ray.tune.schedulers import ASHAScheduler
import optuna
from ray.tune.suggest.optuna import OptunaSearch
from trainers import ProjectedPGTrainer, ProjectedPPOTrainer
from activations import LeakyReLU, Tanh
import torch
import torch.nn as nn
from deq_lib.solvers import broyden, anderson
import os
import math

N_CPUS = 8 # laptop
# N_CPUS  = int(os.getenv('SLURM_CPUS_ON_NODE'))
n_tasks = 1
n_workers_per_task = int(math.floor(N_CPUS/n_tasks))-1-1

# n_workers_per_task = 1

print('==================Using ', n_workers_per_task, ' workers per task==========================')


env = OtherInvertedPendulumEnv
env_config = {  
    "observation": "partial",
    "normed": True,
    "factor": 1,
    # "model_params": torch.load('inv_pend_model_nonlin2.pth')
}

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    # "env": "Taxi-v3",
    "env": env,
    "env_config": env_config,
    "model": {
        "custom_model": ProjRNNOldModel, #tune.grid_search([ProjRENModel, ProjRNNModel, ProjRNNOldModel]),
        "custom_model_config": {
            "state_size": 2,
            "hidden_size": 1,
            "phi_cstor": Tanh, # tune.grid_search([Tanh, LeakyReLU]),
            "log_std_init": np.log(0.2),
            "nn_baseline_n_layers": 2,
            "nn_baseline_size": 64,
            # Projecting controller parameters
            "lmi_eps": 1e-5,
            "exp_stability_rate": 1.0, # 0.9,
            "plant_cstor": env,
            "plant_config": env_config,
            # REN parameters
            "solver": broyden,
            "f_thresh": 30,
            "b_thresh": 30
        }
    },
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    # "num_envs_per_worker": 8,
    "num_workers": n_workers_per_task,
    "framework": "torch",
    "num_gpus": 0,
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
        "explore": False
    },
    "evaluation_interval": 1,
    "evaluation_parallel_to_training": True
}

print('============ Config ==============')
print(config)
print('==================================')

test_env = env(env_config)
print(f'Max reward per: step: {test_env.max_reward}, rollout: {test_env.max_reward*(test_env.time_max+1)}')

ray.init()

def name_creator(trial):
    config = trial.config
    model_cfg = config['model']['custom_model_config']
    name = f"{config['model']['custom_model'].__name__}_{config['env'].__name__}_phi{model_cfg['phi_cstor'].__name__}_state{model_cfg['state_size']}_hidden{model_cfg['hidden_size']}"
    name += f"rho{model_cfg['exp_stability_rate']}"
    return name

results = tune.run(
    ProjectedPPOTrainer,
    config = config,
    stop = {
        # "training_iteration": 200,
        'agent_timesteps_total': 20e3,
    },
    # num_samples = 3,
    # search_alg = optuna_search,
    # scheduler = asha_scheduler,
    # fail_fast = True,
    verbose = 1,
    # verbose = False,
    trial_name_creator = name_creator,
    # name = 'URwd_PPO',
    name = 'scratch',
    # local_dir = '/global/scratch/users/neelayjunnarkar/ray_results',
    local_dir = '../ray_results',
    checkpoint_at_end = True,
    checkpoint_freq = 200,
)

metric = "episode_reward_mean"
best_trial = results.get_best_trial(metric=metric, mode="max", scope="all")
value_best_metric = best_trial.metric_analysis[metric]["max"]
print(
    "Best trial's reward mean(over all "
    "iterations): {}".format(value_best_metric)
)

print('best config: ', results.get_best_config(metric=metric, mode="max"))

# optuna_search = OptunaSearch(metric="episode_reward_mean", mode="max")

# asha_scheduler = ASHAScheduler(
#     time_attr='training_iteration',
#     metric='episode_reward_mean',
#     mode='max',
#     max_t=1000,
#     grace_period=200,
#     reduction_factor=3,
#     brackets=1
# )
# ray.init()
# trainer = ProjectedPGTrainer(env=config["model"]["custom_model_config"]["plant_cstor"], config=config)
# trainer = ProjectedPPOTrainer(env=config["model"]["custom_model_config"]["plant_cstor"], config=config)

# for _ in range(1000):
#     print(f'\n======= Iter {_} =======')
#     trainer.train()
    

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# trainer.evaluate()
