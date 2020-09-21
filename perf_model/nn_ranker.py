import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from .losses import approxNDCGLoss, listMLE


def get_activation(act_type):
    if act_type == 'leaky':
        return nn.LeakyReLU()
    elif act_type == 'elu':
        return nn.ELU()
    else:
        raise NotImplementedError


def get_ranking_loss(loss_type):
    if loss_type == 'approx_ndcg':
        return approxNDCGLoss
    elif loss_type == 'list_mle':
        return listMLE
    else:
        raise NotImplementedError


class RankingModel(nn.Module):
    def __init__(self, in_units, units=128, num_layers=3,
                 dropout=0.1, act_type='leaky'):
        super(RankingModel, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_features=in_units,
                                    out_features=units,
                                    bias=False))
            in_units = units
            layers.append(nn.BatchNorm1d(in_units))
            layers.append(get_activation(act_type))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features=in_units,
                                out_features=1,
                                bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        """

        Parameters
        ----------
        X
            Shape (batch_size, units)

        Returns
        -------
        scores
            Shape (batch_size,)
        """
        return self.net(X)[:, 0]


class RankGroupSampler:
    def __init__(self, thrpt, batch_size=512, group_size=10,
                 beta_params=(3.0, 1.0)):
        self._batch_size = batch_size
        self._num_samples = len(thrpt)
        self._thrpt = thrpt
        self._group_size = group_size
        self._valid_indices = (thrpt > 0).nonzero()[0]
        self._invalid_indices = (thrpt == 0).nonzero()[0]
        # The mixture of dirichlet
        self._beta_params = beta_params

    def __iter__(self):
        """

        Returns
        -------
        indices
            List with shape (batch_size * group_size,)
        """
        while True:
            indices = []
            taus = np.random.beta(a=self._beta_params[0],
                                  b=self._beta_params[1],
                                  size=(self._batch_size,))
            valid_nums = np.ceil(taus * self._group_size).astype(np.int32)
            invalid_nums = self._group_size - valid_nums
            batch_valid_indices = np.random.choice(self._valid_indices, sum(valid_nums),
                                                   replace=True)
            batch_invalid_indices = np.random.choice(self._invalid_indices, sum(invalid_nums),
                                                     replace=True)
            valid_cnt = 0
            invalid_cnt = 0
            for i in range(self._batch_size):
                indices.extend(batch_valid_indices[valid_cnt:(valid_cnt + valid_nums[i])])
                indices.extend(batch_invalid_indices[invalid_cnt:(invalid_cnt + invalid_nums[i])])
                valid_cnt += valid_nums[i]
                invalid_cnt += invalid_nums[i]
            yield indices
