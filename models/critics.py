import chainer
import chainer.links as L
import chainer.functions as F


class _Critic(chainer.Chain):
    def save(self, path):
        if path.exists():
            raise ValueError('File already exist')
        chainer.serializers.save_npz(path.resolve(), self)

    def load(self, path):
        if not path.exists():
            raise ValueError('File {} not found'.format(path))
        chainer.serializers.load_npz(path.resolve(), self)


class MujocoCritic(_Critic):
    def __init__(self, state_dim, action_dim):
        super(MujocoCritic, self).__init__()
        with self.init_scope():
            self._linear1 = L.Linear(
                in_size=(state_dim+action_dim), out_size=400)
            self._linear2 = L.Linear(in_size=400, out_size=300)
            self._linear3 = L.Linear(in_size=300, out_size=1)
        self._state_dim = state_dim
        self._action_dim = action_dim

    def __call__(self, s, a):
        x = F.concat((s, a))
        h = self._linear1(x)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)
        q = self._linear3(h)
        return q
