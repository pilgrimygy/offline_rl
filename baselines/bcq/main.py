import os
import sys
# sys.path.append(os.path.join(os.getcwd(), './'))
# sys.path.append(os.path.join(os.getcwd(), '../..'))
import gym
import numpy as np
import d4rl
import click
from unstable_baselines.common.logger import Logger
from unstable_baselines.offline_rl.baselines.bcq.agent import BCQAgent
from unstable_baselines.offline_rl.baselines.bcq.trainer import BCQTrainer
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed
from unstable_baselines.offline_rl.common.env_wrapper import get_env
from unstable_baselines.common.buffer import ReplayBuffer


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str)
@click.option("--log-dir", default="logs/bcq")
@click.option("--gpu", type=int, default=0)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=35)
@click.option("--info", type=str, default="")
@click.argument('args', nargs=-1)
def main(config_path, log_dir, gpu, print_log, seed, info, args):
    
    args = load_config(config_path, args)

    #set global seed
    set_global_seed(seed)

    #initialize logger
    env_name = args['env_name']
    logger = Logger(log_dir, prefix = env_name+"-"+info, print_to_terminal=print_log)
    logger.log_str("logging to {}".format(logger.log_path))

    #set device and logger
    set_device_and_logger(gpu, logger)

    #save args
    logger.log_str_object("parameters", log_dict = args)

    #initialize environment
    logger.log_str("Initializing Environment")
    env = get_env(env_name)
    eval_env = get_env(env_name)
    state_space = env.observation_space
    action_space = env.action_space

    #initialize dataset and buffer
    logger.log_str("Initializing Dataset") 
    dataset = d4rl.qlearning_dataset(env)

    buffer = ReplayBuffer(state_space, action_space, **args['buffer'])

    buffer.add_traj(dataset['observations'], dataset['actions'], dataset['next_observations'], np.squeeze(dataset['rewards']), np.squeeze(dataset['terminals']))

    logger.log_str("Initializing Agent")
    agent = BCQAgent(state_space, action_space, **args['agent'])

    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer = BCQTrainer(
        agent,
        env,
        eval_env,
        buffer,
        logger,
        **args['trainer']
    )
    
    logger.log_str("Started training")
    trainer.train()


if __name__ == "__main__":
    main()