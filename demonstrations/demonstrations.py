import gymnasium as gym
import torch
import numpy as np
import argparse
from tianshou.data import Batch
import os
from gymnasium.wrappers.record_video import RecordVideo

from ddpg_policy import get_ddpg_policy
from td3_policy import get_td3_policy
from sac_policy import get_sac_policy
from redq_policy import get_redq_policy
from reinforce_policy import get_reinforce_policy
from a2c_policy import get_a2c_policy
from npg_policy import get_npg_policy
from trpo_policy import get_trpo_policy
from ppo_policy import get_ppo_policy

def load_policy(policy_type, env, **kwargs):
    policy_path = kwargs["policy_path"]

    # model
    if policy_type == "ddpg":
        policy, args = get_ddpg_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "td3":
        policy, args = get_td3_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "sac":
        policy, args = get_sac_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "redq":
        policy, args = get_redq_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "reinforce":
        policy, args = get_reinforce_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "a2c":
        policy, args = get_a2c_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "npg":
        policy, args = get_npg_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "trpo":
        policy, args = get_trpo_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "ppo":
        policy, args = get_ppo_policy(env, **kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    else:
        raise Exception("Unknown policy.")

    return policy, args

def simulate(task, policy_type, policy_path, hidden_sizes):

    video_name_prefix = policy_type.upper() + "_" + task
    video_folder = os.path.join("", task, policy_type)

    env = RecordVideo(
        env=gym.make(task, render_mode="rgb_array"),
        video_folder=video_folder,
        name_prefix=video_name_prefix,
        video_length=20000
    )
    observation, info = env.reset()

    policy, args = load_policy(policy_type=policy_type, env=env, task=task, hidden_sizes=hidden_sizes, policy_path=policy_path)
    print("Loaded agent from: ", policy_path)

    reward = -1     # initialize

    for step_index in range(2000):

        batch = Batch(obs=[observation], info=info)  # the first dimension is batch-size
        action = policy.forward(batch=batch, state=observation).act[0].detach().numpy()  # policy.forward return a batch, use ".act" to extract the action

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == '__main__':

    policies = {
        "ddpg",
        "td3",
        "sac",
        "redq",
        "reinforce",
        "a2c",
        "npg",
        "trpo",
        "ppo"
    }

    hidden_sizes = {
        "ddpg": [256,256],
        "td3": [256,256],
        "sac": [256,256],
        "redq": [256,256],
        "reinforce": [64,64],
        "a2c": [64,64],
        "npg": [64,64],
        "trpo": [64,64],
        "ppo": [64,64]
        }

    tasks = [
        "Ant-v4",
        "HalfCheetah-v4",
        "Hopper-v4",
        "Humanoid-v4",
        "HumanoidStandup-v4",
        "InvertedDoublePendulum-v4",
        "InvertedPendulum-v4",
        "Pusher-v4",
        "Reacher-v4",
        "Swimmer-v4",
        "Walker2d-v4"
        ]

    # save all simulations
    for task in tasks:
        for policy_type in policies:

            log_dir = os.path.join("..", "benchmark", "log")
            seed = "0"
            partial_path = os.path.join(log_dir, task, policy_type, seed)
            dir = os.listdir(partial_path)[0]   # there is only one directory
            full_path = os.path.join(partial_path, dir, "policy.pth")

            simulate(
                task=task,
                policy_type=policy_type,
                policy_path=full_path,
                hidden_sizes=hidden_sizes[policy_type]
            )
