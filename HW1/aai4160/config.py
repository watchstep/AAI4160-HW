import os
import wandb
import time
import argparse

from aai4160.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES

def parse_args():
    parser = argparse.ArgumentParser()
    # file path is relative to where you're running this script from
    parser.add_argument('--exp_num', '-en', type=str, required=True)
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)
    parser.add_argument('--expert_data', '-ed', type=str, required=True)
    parser.add_argument('--env_name', '-env', type=str,
        help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str,
        default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int,
        default=1000)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    # training data collected (in the env) during each iteration
    parser.add_argument('--batch_size', type=int, default=1000)
    # eval data collected (in the env) for logging metrics
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    # number of sampled data points to be used per gradient/train step
    parser.add_argument('--train_batch_size', type=int, default=100)

    # depth, of policy to be learned
    parser.add_argument('--n_layers', type=int, default=2)
    # width of each layer, of policy to be learned
    parser.add_argument('--size', type=int, default=64)
    # LR for supervised learning
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--criterion', '-c', type=str, default="MSE")
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    
    return args
    
# sweep_configuration = {
#     "method": "random",
#     "name": "sweep",
#     "metric": {"goal": "maximize", "name": "val_acc"},
#     "parameters": {
#         "batch_size": {"values": [16, 32, 64]},
#         "epochs": {"values": [5, 10, 15]},
#         "lr": {"max": 0.1, "min": 0.0001},
#     },
# }

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="aai4160_hw1")