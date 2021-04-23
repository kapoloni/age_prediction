"""
Trainer module
"""
# Standard library imports
from __future__ import print_function
from __future__ import absolute_import
from collections import OrderedDict
import functools
import math
import numpy as np

# Third party imports
import torch as th
import torch.nn as nn

# Local application imports
from ._utils import (_validate_loss_input,
                     _validate_metric_input,
                     _validate_optimizer_input,
                     _is_tuple_or_list,
                     _parse_num_inputs_and_targets_from_loader,
                     )

from .callbacks import CallbackContainer, History, TQDM
from .metrics import MetricContainer, MetricCallback


class ModuleTrainer(object):

    def __init__(self, model):
        """
        ModelTrainer for high-level training of Pytorch models

        Major Parts
        -----------
        - optimizer(s)
        - loss(es)
        - metrics
        - callbacks
        """
        if not isinstance(model, nn.Module):
            raise ValueError('model argument must \
                inherit from torch.nn.Module')
        self.model = model

        # callbacks
        self._callbacks = []

        # metrics
        self._metrics = []
        self._has_metrics = False

        # losses
        self._loss = None
        self._loss_fn = None

        # other properties
        self._in_train_loop = False
        self._stop_training = False

    def set_loss(self, loss):
        self._loss = loss
        if _is_tuple_or_list(loss):
            self._loss_fn = [_validate_loss_input(lo)
                             for lo in loss]
        else:
            self._loss_fn = _validate_loss_input(loss)

    def set_optimizer(self, optimizer, **kwargs):
        if type(optimizer) is type or isinstance(optimizer, str):
            if 'parameters' in kwargs:
                parameters = kwargs['parameters']
            else:
                parameters = self.model.parameters()

            optimizer = _validate_optimizer_input(optimizer)
            self._optimizer = optimizer(parameters, **kwargs)
        else:
            self._optimizer = optimizer

    def set_callbacks(self, callbacks):
        if not _is_tuple_or_list(callbacks):
            callbacks = [callbacks]
        self._callbacks = [self.history] + callbacks

    def set_metrics(self, metrics):
        metrics = [metrics] if not _is_tuple_or_list(metrics) else metrics
        metrics = [_validate_metric_input(m) for m in metrics]
        self._has_metrics = True
        self._metrics = metrics

    def compile(self,
                optimizer,
                loss,
                callbacks=None,
                metrics=None):
        self.set_optimizer(optimizer)
        self.set_loss(loss)

        self.history = History(self)
        self._callbacks = [self.history]

        if callbacks is not None:
            self.set_callbacks(callbacks)

        if metrics is not None:
            self.set_metrics(metrics)
            self.metric_container = MetricContainer(self._metrics)
        else:
            self.metric_container = []
            self._has_metrics = False

    def fit_loader(self,
                   loader,
                   val_loader=None,
                   num_epoch=100,
                   cuda_device=False,
                   verbose=1):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """
        # ----------------------------------------------------------------------
        num_inputs = loader.dataset.num_inputs
        num_targets = loader.dataset.num_targets
        len_inputs = len(loader.dataset)

        if val_loader is not None:
            num_val_inputs = val_loader.dataset.num_inputs
            num_val_targets = val_loader.dataset.num_targets
            if (num_inputs != num_val_inputs) or \
               (num_targets != num_val_targets):
                raise ValueError('num_inputs != num_val_inputs \
                                  or num_targets != num_val_targets')
        has_val_data = val_loader is not None
        num_batches = int(math.ceil(len_inputs / loader.batch_size))
        # ----------------------------------------------------------------------

        fit_helper = _get_helper(self, num_inputs, num_targets)
        fit_loss_fn = fit_helper.get_partial_loss_fn(self._loss_fn)
        fit_forward_fn = fit_helper.get_partial_forward_fn(self.model.train())

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)

            if self._has_metrics:
                self.metric_container.set_helper(fit_helper)
                tmp_callbacks.append(MetricCallback(self.metric_container))

            callback_container = CallbackContainer(
                self._callbacks+tmp_callbacks)
            callback_container.set_trainer(self)
            callback_container.on_train_begin({'len_inputs': len_inputs,
                                               'num_batches': num_batches,
                                               'num_epoch': num_epoch,
                                               'has_val_data': has_val_data,
                                               'has_metrics':
                                               self._has_metrics,
                                               'metrics':
                                               self.metric_container})
            # If using tensorboard callback
            imgs_batch = False
            tb = False
            for callback in callback_container.callbacks:
                if 'TensorBoardCB' in str(callback):
                    tb = True
                    if callback.imgs_batch:
                        imgs_batch = callback.imgs_batch

            for epoch_idx in range(num_epoch):
                epoch_logs = {}
                callback_container.on_epoch_begin(epoch_idx, epoch_logs)

                loader_iter = iter(loader)
                for batch_idx in range(num_batches):
                    self._in_train_loop = True

                    batch_logs = {}
                    callback_container.on_batch_begin(batch_idx, batch_logs)

                    input_batch, \
                        target_batch = fit_helper.grab_batch_from_loader(
                            loader_iter)

                    batch_logs['batch_size'] = len(input_batch)

                    if cuda_device:
                        input_batch, target_batch = fit_helper.move_to_cuda(
                                                    input_batch, target_batch)

                    # --------TensorBoard-----------#
                    # Quantity of batches to save images
                    if tb and (batch_idx in range(imgs_batch)):
                        callback_container.load_images_batch(input_batch)

                    # ----------Optimization--------#
                    self._optimizer.zero_grad()
                    output_batch = fit_forward_fn(input_batch)
                    loss = fit_loss_fn(output_batch, target_batch)
                    loss.backward()
                    self._optimizer.step()
                    # ------------------------------#

                    if self._has_metrics:
                        metrics_logs = self.metric_container(output_batch,
                                                             target_batch)
                        batch_logs.update(metrics_logs)
                        # self.history.batch_metrics.update(metrics_logs)

                    batch_logs['loss'] = loss.item()
                    callback_container.on_batch_end(batch_idx, batch_logs)

                epoch_logs.update(self.history.batch_metrics)
                if has_val_data:
                    self._in_train_loop = False

                    val_epoch_logs = self.evaluate_loader(val_loader,
                                                          cuda_device,
                                                          verbose)

                    epoch_logs.update(val_epoch_logs)
                    epoch_logs.update(batch_logs)

                    self.history.batch_metrics.update(val_epoch_logs)

                callback_container.on_epoch_end(epoch_idx, epoch_logs)

                if self._stop_training:
                    break
        callback_container.on_train_end()

    def predict_loader(self,
                       loader,
                       cuda_device=False,
                       verbose=1):
        # --------------------------------------------------------
        num_inputs, \
            num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        len_inputs = len(loader.dataset)
        num_batches = int(math.ceil(len_inputs / loader.batch_size))
        # --------------------------------------------------------
        predict_helper = _get_helper(self, num_inputs, num_targets=1)
        pred_forward_fn = predict_helper.get_partial_forward_fn(
            self.model.eval())
        loader_iter = iter(loader)

        for batch_idx in range(num_batches):
            input_batch, _ = predict_helper.grab_batch_from_loader(loader_iter)
            if cuda_device:
                input_batch, _ = predict_helper.move_to_cuda(input_batch, _)

            with th.no_grad():
                output_batch = pred_forward_fn(input_batch)

            if batch_idx == 0:
                len_outputs = 1 if not \
                    _is_tuple_or_list(output_batch) else len(output_batch)
                prediction_lists = [[] for _ in range(len_outputs)]

            if len_outputs == 1:
                prediction_lists[0].append(output_batch)
            else:
                for out_idx in range(len_outputs):
                    prediction_lists[out_idx].append(output_batch[out_idx])

        final_pred_list = [th.cat(pred_list, 0) for
                           pred_list in prediction_lists]

        return final_pred_list if len_outputs > 1 else final_pred_list[0]

    def evaluate_loader(self,
                        loader,
                        cuda_device=False,
                        verbose=1):
        # --------------------------------------------------------
        num_inputs, \
            num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        len_inputs = len(loader.dataset)
        num_batches = int(math.ceil(len_inputs / loader.batch_size))
        # --------------------------------------------------------

        evaluate_helper = _get_helper(self, num_inputs, num_targets)
        eval_loss_fn = evaluate_helper.get_partial_loss_fn(self._loss_fn)
        eval_forward_fn = evaluate_helper.get_partial_forward_fn(
            self.model.eval())
        eval_logs = {'val_loss': 0.}
        loader_iter = iter(loader)
        if self._has_metrics:
            metric_container = MetricContainer(self._metrics, prefix='val_')
            metric_container.set_helper(evaluate_helper)
            metric_container.reset()
            for metric in metric_container.metrics:
                eval_logs.update({'val_' + metric._name: 0.})

        for batch_idx in range(num_batches):
            input_batch, \
                target_batch = evaluate_helper.grab_batch_from_loader(
                    loader_iter)
            if cuda_device:
                input_batch, \
                    target_batch = evaluate_helper.move_to_cuda(input_batch,
                                                                target_batch)

            batch_size = len(input_batch)

            with th.no_grad():
                output_batch = eval_forward_fn(input_batch)
            loss = eval_loss_fn(output_batch, target_batch)

            eval_logs['val_loss'] += loss.item() * batch_size

            if self._has_metrics:
                metrics_logs = metric_container(output_batch, target_batch)
                for metric in metrics_logs:
                    eval_logs[metric] += metrics_logs[metric]*batch_size

        eval_logs['val_loss'] = eval_logs['val_loss'] / len_inputs
        if self._has_metrics:
            for metric in metrics_logs:
                eval_logs[metric] = eval_logs[metric] / len_inputs

        return eval_logs

    def summary(self, input_size, batch_size=-1,
                device=th.device('cuda'), dtypes=None):
        result, params_info = self.summary_string(input_size,
                                                  batch_size,
                                                  device,
                                                  dtypes)
        print(result)

        return params_info

    def summary_string(self, input_size, batch_size=-1,
                       device=th.device('cuda'), dtypes=None):
        if dtypes is None:
            dtypes = [th.FloatTensor]*len(input_size)

        summary_str = ''

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                        ]
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = batch_size

                params = 0
                if hasattr(module, 'weight') and \
                   hasattr(module.weight, "size"):
                    params += th.prod(th.LongTensor(
                        list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias') and \
                   hasattr(module.bias, "size"):
                    params += th.prod(th.LongTensor(
                        list(module.bias.size())))
                summary[m_key]['nb_params'] = params

            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList):
                hooks.append(module.register_forward_hook(hook))

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [th.rand(2, *in_size).type(dtype).to(device=device)
             for in_size, dtype in zip(input_size, dtypes)]

        # create properties
        summary = OrderedDict()
        hooks = []

        # register forward hooks
        self.model.apply(register_hook)

        # make a forward pass
        self.model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        summary_str += 64 * "-" + "\n"
        line_new = "{:>20}  {:>25} {:>15}".format(
            "Layer (type)", "Output Shape", "Param #")
        summary_str += line_new + "\n"
        summary_str += 64 * "=" + "\n"
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]

            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] is True:
                    trainable_params += summary[layer]["nb_params"]
            summary_str += line_new + "\n"

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(sum(input_size, ()))
                               * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. /
                                (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        summary_str += 64 * "=" + "\n"
        summary_str += "Total params: {0:,}".format(total_params) + "\n"
        summary_str += "Trainable params: {0:,}".format(
            trainable_params) + "\n"
        summary_str += "Non-trainable params: {0:,}".format(
            total_params - trainable_params) + "\n"
        summary_str += 64 * "-" + "\n"
        summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
        summary_str += "Forward/backward pass size (MB): \
            %0.2f" % total_output_size + "\n"
        summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
        summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
        summary_str += 64 * "-" + "\n"

        return summary_str, (total_params, trainable_params)

    def save_state_dict(self, file):
        """
        Save a model parameters to disk
        """
        # model parameters -> ordered dict
        state_dict = self.model.state_dict()
        th.save(state_dict, file)


def _get_helper(trainer, num_inputs, num_targets):
    if (num_inputs == 1) and (num_targets == 1):
        helper = SingleInput_SingleTarget_Helper()
    elif (num_inputs == 1) and (num_targets == 0):
        helper = SingleInput_NoTarget_Helper()

    return helper


class SingleInput_SingleTarget_Helper(object):
    def move_to_cuda(self, inputs, targets):
        device = th.device('cuda')
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        return inputs, targets

    def grab_batch_from_loader(self, loader_iter):
        input_batch, target_batch = next(loader_iter)
        return input_batch, target_batch

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = tforms[1](target_batch)
        input_batch, target_batch = tforms[2](input_batch, target_batch)
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class SingleInput_NoTarget_Helper(object):
    def move_to_cuda(self, inputs, targets):
        device = th.device('cuda')
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        return inputs, targets

    def grab_batch_from_loader(self, loader_iter):
        input_batch = next(loader_iter)
        return input_batch, None

    def apply_transforms(self, tforms, input_batch, target_batch=None):
        input_batch = tforms[0](input_batch)
        return input_batch, None

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)
