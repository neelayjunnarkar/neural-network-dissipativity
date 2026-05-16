"""
Configure and train controllers.
"""

import argparse
import math
import multiprocessing
import os

import numpy as np
import ray
from ray import tune

from envs import FlexibleArmEnv, InvertedPendulumEnv, TimeDelayInvertedPendulumEnv
from models import (
    RINN,
    DissipativeSimplestRINN,
    FullyConnectedNetwork,
    ImplicitModel,
    LTIModel,
    SoftDissipativeRINN,
)
from trainers import NonProjectedPPOTrainer, ProjectedPPOTrainer
from utils import ExportWeightsCallback

# =====================
# ENVIRONMENT CONFIGS
# =====================


def get_inverted_pendulum_env(
    seed, saturate_inputs, reward_type=None, reset_scale=1.0, sim_disturbance=False
):
    dt = 0.01
    env = InvertedPendulumEnv
    env_config = {
        "observation": "partial",
        "normed": True,
        "dt": dt,
        "supply_rate": "stability",  # "stability",
        "disturbance_model": "occasional",
        "seed": seed,
        "saturate_inputs": saturate_inputs,
        "reset_scale": reset_scale,
        "sim_disturbance": sim_disturbance,
    }
    if reward_type is not None:
        env_config["reward_type"] = reward_type
    return dt, env, env_config


def get_time_delay_inverted_pendulum_env(seed, saturate_inputs):
    dt = 0.01
    env = TimeDelayInvertedPendulumEnv
    env_config = {
        "observation": "partial",
        "normed": True,
        "dt": dt,
        "design_time_delay": 0.07,
        "time_delay_steps": 5,
        "seed": seed,
        "saturate_inputs": saturate_inputs,
    }
    return dt, env, env_config


def get_flexible_arm_env_full(seed, saturate_inputs, l2_gain=None, reward_type=None):
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
        "seed": seed,
        "saturate_inputs": saturate_inputs,
    }
    if l2_gain is not None:
        env_config["gamma"] = l2_gain
    if reward_type is not None:
        env_config["reward_type"] = reward_type
    return dt, env, env_config


def get_flexible_arm_env_partial(seed, saturate_inputs, l2_gain=None, reward_type=None):
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
        "b": 0.1,  # uncertainty bound ||Delta|| <= b for utox2
        "supplyrate_scale": 0.5,
        "lagrange_multiplier": 5,
        # "design_integrator_type": "utoy",
        # "b": 0.0005,
        # "supplyrate_scale": 1,
        # "lagrange_multiplier": 1000,
        "saturate_inputs": saturate_inputs,
        "seed": seed,
    }
    if l2_gain is not None:
        env_config["gamma"] = l2_gain
    if reward_type is not None:
        env_config["reward_type"] = reward_type
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


def get_rinn_model(dt, env, env_config, nonlin_size):
    return {
        "custom_model": RINN,
        "custom_model_config": {
            "state_size": 2,
            "nonlin_size": nonlin_size,
            "dt": dt,
            "log_std_init": np.log(1.0),
        },
    }


def get_dissipative_simplest_rinn_model(
    dt, env, env_config, trs, backoff, nonlin_size, soft_weight=0.0
):
    return {
        "custom_model": DissipativeSimplestRINN,
        "custom_model_config": {
            "state_size": 2,
            "nonlin_size": nonlin_size,
            "log_std_init": np.log(1.0),
            "dt": dt,
            "plant": env,
            "plant_config": env_config,
            "eps": 1e-3,
            "mode": "thetahat",
            "trs_mode": "fixed",
            "min_trs": trs,
            "backoff_factor": backoff,
            "soft_weight": soft_weight,
            "lti_initializer": "dissipative_thetahat",
            "lti_initializer_kwargs": {
                "trs_mode": "fixed",
                "min_trs": trs,
                "backoff_factor": backoff,
            },
        },
    }


def get_soft_dissipative_rinn_model(
    dt,
    env,
    env_config,
    trs,
    backoff,
    nonlin_size,
    soft_weight,
    free_P,
    free_Lambda,
    nonlin_init_scale=0.0,
    use_lti_init=True,
):
    cfg = {
        "state_size": 2,
        "nonlin_size": nonlin_size,
        "log_std_init": np.log(1.0),
        "dt": dt,
        "plant": env,
        "plant_config": env_config,
        "eps": 1e-3,
        "soft_weight": soft_weight,
        "free_P": free_P,
        "free_Lambda": free_Lambda,
    }
    if use_lti_init:
        cfg.update({
            "nonlin_init_scale": nonlin_init_scale,
            "lti_initializer": "dissipative_thetahat",
            "lti_initializer_kwargs": {
                "trs_mode": "fixed",
                "min_trs": trs,
                "backoff_factor": backoff,
            },
        })
    return {"custom_model": SoftDissipativeRINN, "custom_model_config": cfg}


def get_lti_model(dt, env, env_config, trs, backoff, soft_weight=0.0):
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
            "min_trs": trs,
            "soft_weight": soft_weight,
            "lti_controller": "dissipative_thetahat",
            "lti_controller_kwargs": {
                "trs_mode": "fixed",
                "min_trs": trs,
                "backoff_factor": backoff,
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
        choices=["fcnn", "rinn", "drinn", "lti", "srinn"],
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
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="scratch",
        help="Name of the experiment for logging",
    )
    parser.add_argument(
        "--no_input_saturation",
        action="store_true",
        help="If set, environment will clip actions to action_space bounds (default: False)",
    )
    parser.add_argument(
        "--l2_gain",
        type=float,
        default=None,
        help="L2 gain gamma for flexible arm environments",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default=None,
        help="Reward type for environments (default or quadratic)",
    )
    parser.add_argument(
        "--timesteps_total",
        type=int,
        default=1e7,
        help="Total number of environment timesteps for training",
    )
    parser.add_argument(
        "--nonlin_size",
        type=int,
        default=16,
        help="Number of activation functions (nonlin_size) in RINN models",
    )
    parser.add_argument(
        "--proj_freq",
        type=int,
        default=1,
        help="Apply dissipativity projection every this many gradient steps (default: 1, i.e., every step)",
    )
    parser.add_argument(
        "--soft_weight",
        type=float,
        default=0.0,
        help="Penalty coefficient for soft dissipativity (drinn, lti, and srinn models)",
    )
    parser.add_argument(
        "--free_P",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, P is a learnable parameter in the soft dissipativity model (default: True)",
    )
    parser.add_argument(
        "--free_Lambda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, Lambda is a learnable parameter in the soft dissipativity model (default: True)",
    )
    parser.add_argument(
        "--nonlin_init_scale",
        type=float,
        default=0.0,
        help="Scale for random init of nonlinear paths after LTI init in SoftDissipativeRINN. "
        "0.0 = zero init (original behavior). Recommended: 0.1",
    )
    parser.add_argument(
        "--reset_scale",
        type=float,
        default=1.0,
        help="Scale factor for initial condition range in inverted_pendulum (default: 1.0)",
    )
    parser.add_argument(
        "--sim_disturbance",
        action="store_true",
        help="Apply occasional torque disturbance through Bpu in simulation (does not affect Bpd or dissipativity certificates)",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Save a checkpoint (and export weights) every this many training iterations",
    )
    parser.add_argument(
        "--no_lti_init",
        action="store_true",
        help="If set, SoftDissipativeRINN uses pure RINN random init (no LTI warm-start)",
    )
    args = parser.parse_args()

    args.saturate_inputs = not args.no_input_saturation

    use_savio = False
    if use_savio:
        print("\n\n\nUsing Savio\n===========\n\n\n")
        N_CPUS = int(os.getenv("SLURM_CPUS_ON_NODE"))
        JOB_ID = os.getenv("SLURM_JOB_ID")
    else:
        N_CPUS = multiprocessing.cpu_count()
        if N_CPUS >= 24:
            N_CPUS = 16  # Forcing to smaller number for nicer sharing lab computer
        JOB_ID = None
    n_tasks = 1
    n_workers_per_task = int(math.floor(N_CPUS / n_tasks)) - 2

    # Select environment based on command-line argument
    if args.env == "flexible_rod_partial":
        dt, env, env_config = get_flexible_arm_env_partial(
            args.seed, args.saturate_inputs, args.l2_gain, args.reward_type
        )
    elif args.env == "flexible_rod_full":
        dt, env, env_config = get_flexible_arm_env_full(
            args.seed, args.saturate_inputs, args.l2_gain, args.reward_type
        )
    elif args.env == "inverted_pendulum":
        dt, env, env_config = get_inverted_pendulum_env(
            args.seed,
            args.saturate_inputs,
            args.reward_type,
            reset_scale=args.reset_scale,
            sim_disturbance=args.sim_disturbance,
        )
    elif args.env == "time_delay_inverted_pendulum":
        dt, env, env_config = get_time_delay_inverted_pendulum_env(
            args.seed, args.saturate_inputs
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # Select model based on command-line argument
    if args.model == "fcnn":
        model_config = get_fully_connected_model(dt, env, env_config)
    elif args.model == "rinn":
        model_config = get_rinn_model(dt, env, env_config, args.nonlin_size)
    elif args.model == "drinn":
        model_config = get_dissipative_simplest_rinn_model(
            dt,
            env,
            env_config,
            args.trs,
            args.backoff,
            args.nonlin_size,
            args.soft_weight,
        )
    elif args.model == "lti":
        model_config = get_lti_model(
            dt, env, env_config, args.trs, args.backoff, args.soft_weight
        )
    elif args.model == "srinn":
        model_config = get_soft_dissipative_rinn_model(
            dt,
            env,
            env_config,
            args.trs,
            args.backoff,
            args.nonlin_size,
            args.soft_weight,
            args.free_P,
            args.free_Lambda,
            args.nonlin_init_scale,
            use_lti_init=not args.no_lti_init,
        )
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
        "clip_actions": False,
        "normalize_actions": args.saturate_inputs,
        # "projection_period": args.proj_freq,
        "checkpoint_freq": args.checkpoint_freq,
        "callbacks": ExportWeightsCallback,
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
    _no_projection_models = {"FullyConnectedNetwork", "SoftDissipativeRINN"}
    if model_config["custom_model"].__name__ in _no_projection_models:
        trainer_cls = NonProjectedPPOTrainer
    else:
        trainer_cls = ProjectedPPOTrainer
        config["projection_period"] = args.proj_freq

    ray.init(num_gpus=0)
    tune.run(
        trainer_cls,
        config=config,
        stop={
            "agent_timesteps_total": args.timesteps_total,
        },
        verbose=1,
        trial_name_creator=name_creator,
        name=args.experiment_name,
        local_dir="ray_results",
        checkpoint_at_end=True,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()
