import logging

import pathlib

import numpy as np

import chainer
import chainer.functions as F
from chainer import optimizers
from chainer.dataset import concat_examples


class BCQ(object):
    def __init__(self, critic_builder, perturbator_builder, vae_builder, state_dim, action_dim, *,
                 gamma=0.99, tau=0.5 * 1e-3, lmb=0.75, num_action_samples=10, num_q_ensembles=2, batch_size=100, device=-1):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._q_ensembles = []
        self._target_q_ensembles = []
        self._q_optimizers = []

        for _ in range(num_q_ensembles):
            q_function = critic_builder(state_dim, action_dim)
            target_q_function = critic_builder(state_dim, action_dim)

            q_optimizer = optimizers.Adam()
            q_optimizer.setup(q_function)

            self._q_ensembles.append(q_function)
            self._target_q_ensembles.append(target_q_function)
            self._q_optimizers.append(q_optimizer)

        self._perturbator = perturbator_builder(state_dim, action_dim)
        self._target_perturbator = perturbator_builder(state_dim, action_dim)
        self._perturbator_optimizer = optimizers.Adam()
        self._perturbator_optimizer.setup(self._perturbator)

        self._vae = vae_builder(state_dim, action_dim)
        self._vae_optimizer = optimizers.Adam()
        self._vae_optimizer.setup(self._vae)

        if not device < 0:
            for q_function in self._q_ensembles:
                q_function.to_device(device=device)

            for target_q_function in self._target_q_ensembles:
                target_q_function.to_device(device=device)

            self._perturbator.to_device(device=device)
            self._target_perturbator.to_device(device=device)
            self._vae.to_device(device=device)

        self._gamma = 0.99
        self._tau = tau
        self._lambda = lmb
        self._num_q_ensembles = num_q_ensembles
        self._num_action_samples = num_action_samples
        self._batch_size = batch_size
        self._device = device

        self._initialized = False

    def train(self, iterator, **kwargs):
        if not self._initialized:
            self._initialize_all_networks()
            self._initialized = True

        batch = concat_examples(iterator.next(), device=self._device)
        vae_update_status = self._train_vae(batch)
        q_update_status = self._q_update(batch)
        perturbator_update_status = self._perturbator_update(batch)
        self._update_all_target_networks(tau=self._tau)

        status = {}
        status.update(vae_update_status)
        status.update(q_update_status)
        status.update(perturbator_update_status)

        return status

    def compute_action(self, s):
        with chainer.using_config('enable_backprop', False), chainer.using_config('train', False):
            s = np.float32(s)
            if s.ndim == 1:
                s = np.reshape(s, newshape=(1, ) + s.shape)
            state = chainer.Variable(s)
            if not self._device < 0:
                state.to_gpu()
            s_rep = F.repeat(x=s, repeats=100, axis=0)
            a_rep = self._perturbator(s_rep, self._vae._decode(s_rep))
            max_index = F.argmax(self._q_ensembles[0](s_rep, a_rep), axis=0)
            a = a_rep[max_index]
            if not self._device < 0:
                a.to_cpu()

            if a.shape[0] == 1:
                return np.squeeze(a.data, axis=0)
            else:
                return a.data

    def save_models(self, outdir, prefix):
        for index, q_func in enumerate(self._q_ensembles):
            q_filepath = pathlib.Path(
                outdir, 'q{}_iter-{}'.format(index, prefix))
            q_func.to_cpu()
            q_func.save(q_filepath)
            if not self._device < 0:
                q_func.to_device(device=self._device)

        perturbator_filepath = pathlib.Path(
            outdir, 'perturbator_iter-{}'.format(prefix))
        vae_filepath = pathlib.Path(outdir, 'vae_iter-{}'.format(prefix))

        self._perturbator.to_cpu()
        self._vae.to_cpu()

        self._perturbator.save(perturbator_filepath)
        self._vae.save(vae_filepath)

        if not self._device < 0:
            self._perturbator.to_device(device=self._device)
            self._vae.to_device(device=self._device)

    def load_models(self, q_param_filepaths, perturbator_filepath, vae_filepath):
        for index, q_func in enumerate(self._q_ensembles):
            q_func.to_cpu()
            if q_param_filepaths:
                q_func.load(q_param_filepaths[index])
            if not self._device < 0:
                q_func.to_device(device=self._device)

        self._perturbator.to_cpu()
        self._vae.to_cpu()

        if perturbator_filepath:
            self._perturbator.load(perturbator_filepath)
        if vae_filepath:
            self._vae.load(vae_filepath)

        if not self._device < 0:
            self._perturbator.to_device(device=self._device)
            self._vae.to_device(device=self._device)

    def _train_vae(self, batch):
        status = {}

        (s, a, _, _, _) = batch
        reconstructed_action, mean, ln_var = self._vae((s, a))
        reconstruction_loss = F.mean_squared_error(reconstructed_action, a)
        latent_loss = 0.5 * \
            F.gaussian_kl_divergence(mean, ln_var, reduce='mean')
        vae_loss = reconstruction_loss + latent_loss

        self._vae_optimizer.target.cleargrads()
        vae_loss.backward()
        vae_loss.unchain_backward()
        self._vae_optimizer.update()

        xp = chainer.backend.get_array_module(vae_loss)
        status['vae_loss'] = xp.array(vae_loss.array)
        return status

    def _q_update(self, batch):
        (s, a, _, _, _) = batch
        target_q_value = self._compute_target_q_value(batch)

        for optimizer in self._q_optimizers:
            optimizer.target.cleargrads()

        loss = 0.0
        for q in self._q_ensembles:
            loss += F.mean_squared_error(target_q_value, q(s, a))

        loss.backward()
        loss.unchain_backward()

        for optimizer in self._q_optimizers:
            optimizer.update()

        xp = chainer.backend.get_array_module(loss)
        status = {}
        status['q_loss'] = xp.array(loss.array)
        return status

    def _perturbator_update(self, batch):
        (s, _, _, _, _) = batch

        sampled_actions = self._vae._decode(s)
        perturbed_actions = self._perturbator(s, sampled_actions)

        loss = -F.mean(self._q_ensembles[0](s, perturbed_actions))

        self._perturbator_optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        self._perturbator_optimizer.update()

        status = {}
        xp = chainer.backend.get_array_module(loss)
        status['perturbator loss'] = xp.array(loss.array)
        return status

    def _compute_target_q_value(self, batch):
        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            (_, _, r, s_next, non_terminal) = batch
            r = F.reshape(r, shape=(*r.shape, 1))
            non_terminal = F.reshape(
                non_terminal, shape=(*non_terminal.shape, 1))

            s_next_rep = F.repeat(
                x=s_next, repeats=self._num_action_samples, axis=0)
            a_next_rep = self._vae._decode(s_next_rep)
            perturbed_action = self._target_perturbator._sample(
                s_next_rep, a_next_rep)
            q_values = F.stack([q_target(s_next_rep, perturbed_action)
                                for q_target in self._target_q_ensembles])
            assert q_values.shape == (
                self._num_q_ensembles, self._batch_size * self._num_action_samples, 1)

            weighted_q_minmax = self._lambda * F.min(q_values, axis=0) \
                + (1 - self._lambda) * F.max(q_values, axis=0)
            assert weighted_q_minmax.shape == (
                self._batch_size * self._num_action_samples, 1)
            next_q_value = F.max(
                F.reshape(weighted_q_minmax, shape=(self._batch_size, -1)), axis=1, keepdims=True)
            assert next_q_value.shape == (self._batch_size, 1)
            target_q_value = r + self._gamma * next_q_value * non_terminal
            target_q_value.unchain()
            assert target_q_value.shape == (self._batch_size, 1)
        return target_q_value

    def _initialize_all_networks(self):
        self._update_all_target_networks(tau=1.0)

    def _update_all_target_networks(self, tau):
        for target_q, q in zip(self._target_q_ensembles, self._q_ensembles):
            self._update_target_network(target_q, q, tau)
        self._update_target_network(
            self._target_perturbator, self._perturbator, tau)

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.params(), origin.params()):
            target_param.data = tau * origin_param.data + \
                (1.0 - tau) * target_param.data
