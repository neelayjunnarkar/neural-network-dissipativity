"""
Configure and train controllers.
"""

import math
import os
import multiprocessing
import argparse

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from envs import FlexibleArmEnv, InvertedPendulumEnv, TimeDelayInvertedPendulumEnv
from models import (
    RINN,
    DissipativeSimplestRINN,
    FullyConnectedNetwork,
    ImplicitModel,
    LTIModel,
)
from trainers import ProjectedPPOTrainer

# =====================
# ENVIRONMENT CONFIGS
# =====================


def get_inverted_pendulum_env(seed):
    dt = 0.01
    env = InvertedPendulumEnv
    env_config = {
        "observation": "partial",
        "normed": True,
        "dt": dt,
        "supply_rate": "stability",  # "stability",
        "disturbance_model": "occasional",
        "saturate_inputs": False,
        "seed": seed,
    }
    return dt, env, env_config


def get_time_delay_inverted_pendulum_env(seed):
    dt = 0.01
    env = TimeDelayInvertedPendulumEnv
    env_config = {
        "observation": "partial",
        "normed": True,
        "dt": dt,
        "design_time_delay": 0.07,
        "time_delay_steps": 5,
        "saturate_inputs": False,
        "seed": seed,
    }
    return dt, env, env_config


def get_flexible_arm_env_full(seed):
    dt = 0.001  # or 0.0001
    env = FlexibleArmEnv
    env_config = {
        "observation": "full",
        "normed": True,
        "dt": dt,
        "rollout_length": int(2.5 / dt) - 1,  # 10000,
        "supply_rate": "l2_gain",
        "disturbance_model": "none",
        "disturbance_design_model": "occasional",
        "design_model": "rigid",
        "saturate_inputs": False,
        "seed": seed,
    }
    return dt, env, env_config


def get_flexible_arm_env_partial(seed):
    dt = 0.001
    env = FlexibleArmEnv
    env_config = {
        "observation": "partial",
        "normed": True,
        "dt": dt,
        "rollout_length": int(2 / dt) - 1,
        "supply_rate": "l2_gain",
        "disturbance_model": "occasional",
        "disturbance_design_model": "occasional",
        "design_model": "rigidplus_integrator",
        "delta_alpha": 1.0,
        "design_integrator_type": "utox2",
        "supplyrate_scale": 0.5,
        "lagrange_multiplier": 5,
        # "design_integrator_type": "utoy",
        # "supplyrate_scale": 1,
        # "lagrange_multiplier": 1000,
        "saturate_inputs": False,
        "seed": seed,
    }
    return dt, env, env_config


# =====================
# MODEL CONFIGS
# =====================


def get_fully_connected_model(dt, env, env_config):
    return {
        "custom_model": FullyConnectedNetwork,
        "custom_model_config": {"n_layers": 2, "size": 19},
    }


def get_implicit_model(dt, env, env_config):
    return {
        "custom_model": ImplicitModel,
        "custom_model_config": {"state_size": 16},
    }


def get_rinn_model(dt, env, env_config):
    return {
        "custom_model": RINN,
        "custom_model_config": {
            "state_size": 2,
            "nonlin_size": 16,
            "dt": dt,
            "log_std_init": np.log(1.0),
        },
    }


def get_dissipative_simplest_rinn_model(dt, env, env_config, trs, backoff):
    return {
        "custom_model": DissipativeSimplestRINN,
        "custom_model_config": {
            "state_size": 2,
            "nonlin_size": 16,
            "log_std_init": np.log(1.0),
            "dt": dt,
            "plant": env,
            "plant_config": env_config,
            "eps": 1e-3,
            "mode": "thetahat",
            "trs_mode": "fixed",
            "min_trs": trs,  # 1,
            "backoff_factor": backoff,  # 1.05,
            "lti_initializer": "dissipative_thetahat",
            "lti_initializer_kwargs": {
                "trs_mode": "fixed",
                "min_trs": trs,  # 1,
                "backoff_factor": backoff,  # 1.05,
            },
        },
    }


def get_lti_model(dt, env, env_config, trs, backoff):
    return {
        "custom_model": LTIModel,
        "custom_model_config": {
            "dt": dt,
            "plant": env,
            "plant_config": env_config,
            "learn": True,
            "log_std_init": np.log(1.0),
            "state_size": 2,
            "trs_mode": "fixed",
            "min_trs": trs, # 1.5, # 1.44,
            "lti_controller": "dissipative_thetahat",
            "lti_controller_kwargs": {
                "trs_mode": "fixed",
                "min_trs": trs,  # 1.5 # 1.44
                "backoff_factor": backoff,  # 1.05
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train controller with selectable model, environment, seed, and learning rate."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["fcnn", "rinn", "drinn", "lti"],
        help="Controller model to use",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=[
            "flexible_rod_partial",
            "flexible_rod_full",
            "inverted_pendulum",
            "time_delay_inverted_pendulum",
        ],
        help="Environment to use",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument(
        "--trs",
        type=float,
        default=1.0,
        help="Factor for numerical conditioning in dissipative models",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.05,
        help="Suboptimal projection allowance for numerical conditioning in dissipative models",
    )
    parser.add_argument("--experiment_name", type=str, default="scratch", help="Name of the experiment for logging")
    args = parser.parse_args()

    use_savio = False
    if use_savio:
        print("\n\n\nUsing Savio\n===========\n\n\n")
        N_CPUS = int(os.getenv("SLURM_CPUS_ON_NODE"))
        JOB_ID = os.getenv("SLURM_JOB_ID")
    else:
        N_CPUS = multiprocessing.cpu_count()
        JOB_ID = None
    n_tasks = 1
    n_workers_per_task = int(math.floor(N_CPUS / n_tasks)) - 2

    # Select environment based on command-line argument
    if args.env == "flexible_rod_partial":
        dt, env, env_config = get_flexible_arm_env_partial(args.seed)
    elif args.env == "flexible_rod_full":
        dt, env, env_config = get_flexible_arm_env_full(args.seed)
    elif args.env == "inverted_pendulum":
        dt, env, env_config = get_inverted_pendulum_env(args.seed)
    elif args.env == "time_delay_inverted_pendulum":
        dt, env, env_config = get_time_delay_inverted_pendulum_env(args.seed)
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # Select model based on command-line argument
    if args.model == "fcnn":
        model_config = get_fully_connected_model(dt, env, env_config)
    elif args.model == "rinn":
        model_config = get_rinn_model(dt, env, env_config)
    elif args.model == "drinn":
        model_config = get_dissipative_simplest_rinn_model(
            dt, env, env_config, args.trs, args.backoff
        )
    elif args.model == "lti":
        model_config = get_lti_model(dt, env, env_config, args.trs, args.backoff)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    config = {
        "env": env,
        "env_config": env_config,
        "model": model_config,
        "sgd_minibatch_size": 2048,
        "train_batch_size": 20480,
        "lr": args.lr,
        "num_envs_per_worker": 10,
        "seed": args.seed,
        "num_workers": n_workers_per_task,
        "framework": "torch",
        "num_gpus": 0,  # 1,
        "evaluation_num_workers": 1,
        "evaluation_config": {"render_env": False, "explore": False},
        "evaluation_interval": 1,
        "evaluation_parallel_to_training": True,
    }

    print("==================================")
    print("Number of workers per task: ", n_workers_per_task)
    print("")
    print("Config: ")
    print(config)
    print("")

    test_env = env(env_config)
    print(
        f"Max reward per: step: {test_env.max_reward}, rollout: {test_env.max_reward * (test_env.time_max + 1)}"
    )
    print("==================================")

    def name_creator(trial):
        config = trial.config
        name = f"{config['env'].__name__}"
        name += f"_{config['model']['custom_model'].__name__}"
        if use_savio:
            name += f"_{JOB_ID}"
        name += f"_seed{args.seed}"
        return name

    # Select trainer based on model
    if model_config["custom_model"].__name__ == "FullyConnectedNetwork":
        trainer_cls = PPOTrainer
    else:
        trainer_cls = ProjectedPPOTrainer

    ray.init()
    tune.run(
        trainer_cls,
        config=config,
        stop={
            "agent_timesteps_total": 1e7,  # 6100e3,
        },
        verbose=1,
        trial_name_creator=name_creator,
        name=args.experiment_name,
        local_dir="ray_results",
        checkpoint_at_end=True,
        checkpoint_freq=1000,
    )


if __name__ == "__main__":
    main()
