import pathlib

import gym

import argparse
import datetime
import os
import sys
import json
import pickle
import time

import numpy as np

import chainer

from bcq import BCQ
from models.actors import VAEActor, MujocoActionPerturbator
from models.critics import MujocoCritic

from wrappers import NumpyFloat32Env, ScreenRenderEnv, EveryEpisodeMonitor

from tensorboardX import SummaryWriter


def build_env(args, seed=None):
    env = gym.make(args.env)
    env = NumpyFloat32Env(env)
    env.seed(seed)
    return env


def write_text_to_file(file_path, data):
    with open(file_path, 'w') as f:
        f.write(data)


def create_dir_if_not_exist(outdir):
    if os.path.exists(outdir):
        if not os.path.isdir(outdir):
            raise RuntimeError('{} is not a directory'.format(outdir))
        else:
            return
    os.makedirs(outdir)


def prepare_output_dir(base_dir, args, time_format='%Y-%m-%d-%H%M%S'):
    time_str = datetime.datetime.now().strftime(time_format)
    outdir = os.path.join(base_dir, time_str)
    create_dir_if_not_exist(outdir)

    # Save all the arguments
    args_file_path = os.path.join(outdir, 'args.txt')
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    write_text_to_file(args_file_path, json.dumps(args))

    # Save the command
    argv_file_path = os.path.join(outdir, 'command.txt')
    argv = ' '.join(sys.argv)
    write_text_to_file(argv_file_path, argv)

    return outdir


def perturbator_builder(state_dim, action_dim):
    return MujocoActionPerturbator(state_dim=state_dim, action_dim=action_dim)


def critic_builder(state_dim, action_dim):
    return MujocoCritic(state_dim=state_dim, action_dim=action_dim)


def vae_builder(state_dim, action_dim):
    return VAEActor(state_dim=state_dim, action_dim=action_dim, latent_dim=action_dim * 2)


def load_data_as_replay_buffer(datafile):
    with open(datafile, 'rb') as f:
        return pickle.load(f)


def start_training(args):
    env = build_env(args)
    test_env = build_env(args, seed=100)

    bcq = BCQ(
        critic_builder=critic_builder,
        perturbator_builder=perturbator_builder,
        vae_builder=vae_builder,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        gamma=args.gamma,
        tau=args.tau,
        lmb=args.lmb,
        num_q_ensembles=args.num_ensembles,
        num_action_samples=args.action_samples,
        batch_size=args.batch_size,
        device=args.gpu)
    load_params(bcq, args)

    replay_buffer = load_data_as_replay_buffer(args.datafile)

    outdir = prepare_output_dir(args.outdir, args)
    summarydir = outdir + '/' + 'summary'
    writer = SummaryWriter(logdir=summarydir)

    iterator = chainer.iterators.SerialIterator(
        replay_buffer, batch_size=args.batch_size)
    for timestep in range(args.total_timesteps):
        status = bcq.train(iterator)
        if timestep % args.evaluation_interval == 0 and timestep != 0:
            _, mean, median, _ = evaluate_policy(test_env, bcq)
            print('mean: {mean}, median: {median}'.format(
                mean=mean, median=median))

            bcq.save_models(outdir, prefix=str(timestep))
            writer.add_scalars(
                'eval_result', {'mean': mean, 'median': median}, global_step=timestep)

            for (key, value) in status.items():
                print('key: ', key, ' value: ', value)
                writer.add_scalar(key, value.get(), global_step=timestep)
    env.close()
    test_env.close()


def start_test_run(args):
    test_env = build_env(args)
    if args.render:
        if not args.save_video:
            test_env = ScreenRenderEnv(test_env)
    if args.save_video:
        test_env = EveryEpisodeMonitor(test_env,
                                       directory='video',
                                       write_upon_reset=True,
                                       force=True,
                                       resume=True,
                                       mode='evaluation')

    bcq = BCQ(
        critic_builder=critic_builder,
        perturbator_builder=perturbator_builder,
        vae_builder=vae_builder,
        state_dim=test_env.observation_space.shape[0],
        action_dim=test_env.action_space.shape[0],
        device=args.gpu)
    load_params(bcq, args)

    _, mean, median, _ = evaluate_policy(test_env, bcq)
    print('mean: {mean}, median: {median}'.format(
        mean=mean, median=median))
    test_env.close()


def evaluate_policy(env, algorithm, *, n_runs=10):
    returns = []
    for _ in range(n_runs):
        s = env.reset()
        episode_return = 0
        while True:
            a = algorithm.compute_action(s)
            time.sleep(.002)  # Workaround to avoid time exception
            s, r, done, _ = env.step(a)
            episode_return += r
            if done:
                returns.append(episode_return)
                break
    return returns, np.mean(returns), np.median(returns), np.std(returns)


def load_params(bear, args):
    print('loading model params')
    q_params = [pathlib.Path(q_params)
                for q_params in args.q_params] if args.q_params else None
    pi_params = pathlib.Path(args.pi_params) if args.pi_params else None
    vae_params = pathlib.Path(args.vae_params) if args.vae_params else None
    lagrange_params = pathlib.Path(
        args.lagrange_params) if args.lagrange_params else None
    bear.load_models(q_params, pi_params, vae_params, lagrange_params)


def main():
    parser = argparse.ArgumentParser()

    # output
    parser.add_argument('--outdir', type=str, default='results')

    # Environment
    parser.add_argument('--env', type=str, default='Walker2d-v2')

    # Gpu
    parser.add_argument('--gpu', type=int, default=-1)

    # training data
    parser.add_argument('--datafile', type=str, required=True)

    # testing
    parser.add_argument('--test-run', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--render', action='store_true')

    # params
    parser.add_argument('--q-params', type=str, nargs="+")
    parser.add_argument('--perturbator-params', type=str, default="")
    parser.add_argument('--vae-params', type=str, default="")

    # Training parameters
    parser.add_argument('--total-timesteps', type=float, default=1000000)
    parser.add_argument('--learning-rate', type=float, default=1.0 * 1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--num-ensembles', type=int, default=2)
    parser.add_argument('--action-samples', type=int, default=10)
    parser.add_argument('--lmb', type=float, default=0.75)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--evaluation-interval', type=float, default=5000)

    args = parser.parse_args()

    if args.test_run:
        start_test_run(args)
    else:
        start_training(args)


if __name__ == "__main__":
    main()
