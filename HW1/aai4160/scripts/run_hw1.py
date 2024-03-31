"""
Runs behavior cloning and DAgger for homework 1
Hyperparameters for the experiment are defined in main()
"""

import os
import wandb
import time
import argparse

from aai4160.infrastructure.bc_trainer import BCTrainer
from aai4160.agents.bc_agent import BCAgent
from aai4160.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from aai4160.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES

os.environ["WANDB_API_KEY"] = "0b727f9817c68e9ff062ee720759d799754c60a1"

def init_wandb():
    wandb.init(
        name="exp_1",
        project="aai4160_hw1",
        tags=["BC"],
        sync_tensorboard=True,
    )
    
def run_bc(params):
    """
    Runs behavior cloning with the specified parameters

    Args:
        params: experiment parameters
    """

    #######################
    ## AGENT PARAMS
    #######################

    agent_params = {
        'n_layers': params['n_layers'],
        'size': params['size'],
        'learning_rate': params['learning_rate'],
        'max_replay_buffer_size': params['max_replay_buffer_size'],
    }
    params['agent_class'] = BCAgent
    params['agent_params'] = agent_params

    #######################
    ## ENVIRONMENT PARAMS
    #######################

    params["env_kwargs"] = MJ_ENV_KWARGS[params['env_name']]

    #######################
    ## LOAD EXPERT POLICY
    #######################

    print('Loading expert policy from...', params['expert_policy_file'])
    loaded_expert_policy = LoadedGaussianPolicy(
        params['expert_policy_file'])
    print('Done restoring expert policy...')

    ###################
    ### RUN TRAINING
    ###################

    trainer = BCTrainer(params)
    trainer.run_training_loop(
        n_iter=params['n_iter'],
        initial_expertdata=params['expert_data'],
        collect_policy=trainer.agent.actor,
        eval_policy=trainer.agent.actor,
        relabel_with_expert=params['do_dagger'],
        expert_policy=loaded_expert_policy,
    )

def main():
    """
    Parses arguments, creates logger, and runs behavior cloning
    """

    parser = argparse.ArgumentParser()
    # file path is relative to where you're running this script from
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
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) \
            of training, to iteratively query the expert and train \
            (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data \
            just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        '../../data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + \
        time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    init_wandb()
    run_bc(params)
    wandb.finish()
    
if __name__ == "__main__":
    main()
