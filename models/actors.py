import chainer
import chainer.functions as F
import chainer.links as L


class _Actor(chainer.Chain):
    def save(self, path):
        if path.exists():
            raise ValueError('File already exist')
        chainer.serializers.save_npz(path.resolve(), self)

    def load(self, path):
        if not path.exists():
            raise ValueError('File {} not found'.format(path))
        chainer.serializers.load_npz(path.resolve(), self)


class MujocoActionPerturbator(_Actor):
    def __init__(self, state_dim, action_dim, phi=0.05):
        super(MujocoActionPerturbator, self).__init__()
        with self.init_scope():
            self._linear1 = L.Linear(
                in_size=(state_dim+action_dim), out_size=400)
            self._linear2 = L.Linear(in_size=400, out_size=300)
            self._linear3 = L.Linear(in_size=300, out_size=action_dim)

        self._phi = phi
        self._action_dim = action_dim

    def __call__(self, state, action):
        h = self._linear1(F.concat((state, action)))
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)
        h = self._linear3(h)
        perturbation = F.tanh(h) * self._phi
        return F.clip(action + perturbation, -1, 1)


class VAEActor(_Actor):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(VAEActor, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._latent_dim = latent_dim
        with self.init_scope():
            self._linear1 = L.Linear(
                in_size=(state_dim + action_dim), out_size=750)
            self._linear2 = L.Linear(in_size=750, out_size=750)

            self._linear_mean = L.Linear(in_size=750, out_size=latent_dim)
            self._linear_ln_var = L.Linear(in_size=750, out_size=latent_dim)

            self._linear3 = L.Linear(
                in_size=(state_dim + latent_dim), out_size=750)
            self._linear4 = L.Linear(in_size=750, out_size=750)
            self._linear5 = L.Linear(in_size=750, out_size=action_dim)

    def __call__(self, x):
        (s, a) = x
        z, mu, ln_var = self._encode(s, a)
        reconstructed, _ = self._decode(s, z)
        return reconstructed, mu, ln_var

    def _encode(self, s, a):
        mu, ln_var = self._latent_distribution(s, a)
        return F.gaussian(mu, ln_var), mu, ln_var

    def _decode(self, s, z=None):
        if z is None:
            xp = chainer.backend.get_array_module(s)
            z = chainer.Variable(xp.random.normal(
                0, 1, size=(s.shape[0], self._latent_dim)))
            z = F.cast(z, typ=xp.float32)
            z = F.clip(z, -0.5, 0.5)
        x = F.concat((s, z), axis=1)
        h = self._linear3(x)
        h = F.relu(h)
        h = self._linear4(h)
        h = F.relu(h)
        h = self._linear5(h)

        return F.tanh(h)

    def _latent_distribution(self, s, a):
        x = F.concat((s, a), axis=1)
        h = self._linear1(x)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)

        mu = self._linear_mean(h)
        ln_var = self._linear_ln_var(h)

        return mu, ln_var
