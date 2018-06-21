# qfunctions.py

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl


class QFunction3LP(chainer.Chain):
    """
    Q^*をDNN(3LP)を使って近似する
    """
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, obs, test=False):
        h = F.tanh(self.l0(obs))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


class MyQFunction(chainer.Chain):
    """
    Q^*をDNNを使って近似する（別実装）
    """
    def __init__(self, obs_size, n_actions):
        # TODO: 必要なら引数を追加して、QFunction3LP の実装を参考にネットワークを構築する
        raise NotImplementedError

    def __call__(self, obs, test=False):
        # TODO: QFunction3LP の実装を参考にネットワーク適用を実装する
        raise NotImplementedError
