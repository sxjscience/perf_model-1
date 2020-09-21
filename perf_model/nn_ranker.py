import torch as th
import torch.nn as nn
import torch.nn.functional as F



class RankingModel(nn.Module):
    def __init__(self, in_units, units=64, num_layers=3,
                 dropout=0.05):
        super(RankingModel, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_features=in_units,
                                    out_features=units))
            in_units = units
            layers.append(nn.BatchNorm1d(in_units))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features=in_units,
                                out_features=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RankGroupSampler:
    def __init__(self, num_samples, batch_size=512, group_size=10):
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._group_size = group_size
        self._distribution = th.ones((batch_size, num_samples)) / num_samples

    def __iter__(self):
        with th.no_grad():
            indices = th.multinomial(self._distribution, num_samples=self._group_size,
                                     replacement=False)
