"""
Utility functions for the trainer module
"""
# Third party imports
import torch.nn.functional as F
import torch.optim as optim

# Local application imports
from age_prediction.metrics import (Metric,
                                    CategoricalAccuracy,
                                    BinaryAccuracy,
                                    MSE,
                                    MAE)


def _is_iterable(x):
    return isinstance(x, (tuple, list))


def _is_tuple_or_list(x):
    return isinstance(x, (tuple, list))


def _parse_num_inputs_and_targets_from_loader(loader):

    num_inputs = loader.dataset.num_inputs
    num_targets = loader.dataset.num_targets
    return num_inputs, num_targets


def _validate_metric_input(metric):
    if isinstance(metric, str):
        if metric.upper() == 'CATEGORICAL_ACCURACY' or \
           metric.upper() == 'ACCURACY':
            return CategoricalAccuracy()
        elif metric.upper() == 'BINARY_ACCURACY':
            return BinaryAccuracy()
        elif metric.upper() == 'MSE':
            return MSE()
        elif metric.upper() == 'MAE':
            return MAE()
        else:
            raise ValueError('Invalid metric string input -\
                              must match pytorch function.')
    elif isinstance(metric, Metric):
        return metric
    else:
        raise ValueError('Invalid metric input')


def _validate_loss_input(loss):
    dir_f = dir(F)
    loss_fns = [d.lower() for d in dir_f]
    if isinstance(loss, str):
        if loss.lower() == 'unconstrained':
            return lambda x: x
        elif loss.lower() == 'unconstrained_sum':
            return lambda x: x.sum()
        elif loss.lower() == 'unconstrained_mean':
            return lambda x: x.mean()

        else:
            try:
                str_idx = loss_fns.index(loss.lower())
            except 0:
                raise ValueError('Invalid loss string input - \
                                  must match pytorch function.\
                                  Available loss', loss_fns)
            return getattr(F, dir(F)[str_idx])
    elif callable(loss):
        return loss
    else:
        raise ValueError('Invalid loss input')


def _validate_optimizer_input(optimizer):
    dir_optim = dir(optim)
    opts = [o.lower() for o in dir_optim]
    if isinstance(optimizer, str):
        try:
            str_idx = opts.index(optimizer.lower())
        except 0:
            raise ValueError('Invalid optimizer string input - \
                              must match pytorch function.')
        return getattr(optim, dir_optim[str_idx])
    elif hasattr(optimizer, 'step') and hasattr(optimizer, 'zero_grad'):
        return optimizer
    else:
        raise ValueError('Invalid optimizer input')
