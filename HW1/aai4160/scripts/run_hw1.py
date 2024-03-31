"""
Runs behavior cloning and DAgger for homework 1
Hyperparameters for the experiment are defined in main()
"""

import os
import wandb
import time

from aai4160.infrastructure.bc_trainer import BCTrainer
from aai4160.agents.bc_agent import BCAgent
from aai4160.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from aai4160.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES
from aai4160.config import parse_args

os.environ["WANDB_API_KEY"] = "0b727f9817c68e9ff062ee720759d799754c60a1"

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
       
    args = parse_args()
    
    wandb.init(
        name=f"exp_{args.exp_num}",
        project="aai4160_hw1",
        tags=["BC"],
        sync_tensorboard=True,
        config=args,
    )


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
    
    run_bc(params)
    wandb.finish()
    
if __name__ == "__main__":
    main()
