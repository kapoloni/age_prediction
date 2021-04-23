"""
Metrics module
"""
# Standard library imports
import numpy as np

# Third party imports
import torch as th
from torch import nn

# Local application imports
from age_prediction.callbacks import Callback


class MetricContainer(object):

    def __init__(self, metrics, prefix=''):
        self.metrics = metrics
        self.helper = None
        self.prefix = prefix

    def set_helper(self, helper):
        self.helper = helper

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def __call__(self, output_batch, target_batch):
        logs = {}
        for metric in self.metrics:
            logs[self.prefix +
                 metric._name] = self.helper.calculate_loss(output_batch,
                                                            target_batch,
                                                            metric)

        return logs


class Metric(object):

    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Custom Metrics must \
            implement this function')

    def reset(self):
        raise NotImplementedError('Custom Metrics must \
            implement this function')


class MetricCallback(Callback):

    def __init__(self, container):
        self.container = container

    def on_epoch_begin(self, epoch_idx, logs):
        self.container.reset()


class CategoricalAccuracy(Metric):

    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

    def __call__(self, y_pred, y_true):
        top_k = y_pred.topk(self.top_k, 1)[1]
        true_k = y_true.view(len(y_true), 1).expand_as(top_k)
        self.correct_count += top_k.eq(true_k).float().sum().item()
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy


def bin_centers():
    bin_range = [20, 95]
    bin_step = 1
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    bin_number = int(bin_length / bin_step)
    return bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)


def is_tuple_or_list_of_list(output_batch, target_batch):
    if output_batch[0].shape[0] > 1:
        bc = bin_centers()
        output_batch = (np.exp(output_batch.detach().cpu())@bc)
        target_batch = (target_batch.detach().cpu()@bc)
        return output_batch, target_batch
    else:
        return output_batch, target_batch


class MSE(Metric):
    def __init__(self):
        self._name = 'mse_metric'
        self.mse = 0

    def reset(self):
        # self.history.batch_metrics[self._name] = 0.
        self.mse = 0
        self.n = 0

    def __call__(self, y_pred, y_true):
        y_pred, y_true = is_tuple_or_list_of_list(y_pred, y_true)
        if not th.is_tensor(y_pred) and not th.is_tensor(y_true):
            y_pred, y_true = th.from_numpy(y_pred), th.from_numpy(y_true)
        self.n += y_pred.numel()
        self.mse = nn.MSELoss()(y_pred, y_true).cpu().detach().numpy().item()
        return self.mse

    def value(self):
        return self.mse*self.n, self.n


class MAE(Metric):
    def __init__(self):
        self._name = 'mae_metric'
        self.mae = 0

    def reset(self):
        # self.history.batch_metrics[self._name] = 0.
        self.mae = 0

    def __call__(self, y_pred, y_true):
        y_pred, y_true = is_tuple_or_list_of_list(y_pred, y_true)
        self.mae = nn.L1Loss(reduction='mean'
                             )(y_pred, y_true).cpu().detach().numpy().item()
        return self.mae


class BinaryAccuracy(Metric):

    def __init__(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

    def __call__(self, y_pred, y_true):
        y_pred_round = y_pred.round().long()
        self.correct_count += y_pred_round.eq(y_true).float().sum().item()
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy
