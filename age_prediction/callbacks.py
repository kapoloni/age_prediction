"""
SuperModule Callbacks
"""

from __future__ import absolute_import
from __future__ import print_function

from collections import OrderedDict
from collections import Iterable

import os
import csv
import shutil
import math

from tqdm import tqdm
import numpy as np
import torch as th

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from PIL import Image
from skimage.exposure import rescale_intensity
# from ._utils import (_get_current_time)
import datetime


def _get_current_time(strft=False):
    if strft:
        return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")
    else:
        return datetime.datetime.now()


class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """
    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs['start_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        logs['final_loss'] = self.trainer.history['loss'][-1],
        logs['best_loss'] = min(self.trainer.history['loss']),
        logs['stop_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def load_images_batch(self, batch_imgs):
        self.batch_imgs = batch_imgs
        for callback in self.callbacks:
            if 'TensorBoardCB' in str(callback):
                callback.set_batch_images(batch_imgs)


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_batch_images(self, batch_imgs):
        self.batch_imgs = batch_imgs

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class TQDM(Callback):

    def __init__(self):
        """
        TQDM Progress Bar callback

        This callback is automatically applied to
        every SuperModule if verbose > 0
        """
        self.progbar = None
        super(TQDM, self).__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar is not None:
            self.progbar.close()

    def on_train_begin(self, logs):
        self.train_logs = logs

    def on_epoch_begin(self, epoch, logs=None):
        try:
            self.samples_seen = 0.
            self.progbar = tqdm(total=self.train_logs['num_batches'],
                                unit=' batches')
            self.progbar.set_description('Epoch %i/%i' %
                                         (epoch+1,
                                          self.train_logs['num_epoch']))
        except ValueError:
            pass

    def on_epoch_end(self, epoch, logs=None):
        log_data = {key: '%.04f' % value[epoch] for key, value in
                    self.trainer.history.epoch_metrics.items()}

        for k, v in logs.items():
            if k.endswith('metric'):
                if k not in list(log_data.keys()):
                    log_data[k.split('_metric')[0]] = '%.02f' % v
                else:
                    log_data[k.split('_metric')[0]] = log_data[k]
                    del log_data[k]

        self.progbar.set_postfix(log_data)
        # self.progbar.update(1)
        self.progbar.close()

    def on_batch_begin(self, batch, logs=None):
        self.progbar.update(1)

    def on_batch_end(self, batch, logs=None):
        self.samples_seen += logs['batch_size']

        log_data = {key: value/self.samples_seen for key, value in
                    self.trainer.history.batch_metrics.items()
                    if 'val_' not in key}

        log_data = {key: '%.04f' % value for key, value in
                    log_data.items()}

        for k, v in logs.items():
            if 'val_' not in k:
                if k.endswith('metric'):
                    if k not in list(log_data.keys()):
                        log_data[k.split('_metric')[0]] = '%.02f' % v
                    else:
                        log_data[k.split('_metric')[0]] = log_data[k]
                        del log_data[k]

        self.progbar.set_postfix(log_data)


class History(Callback):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every SuperModule.
    """
    def __init__(self, model):
        super(History, self).__init__()
        self.samples_seen = 0.
        self.trainer = model

    def on_train_begin(self, logs=None):
        self.epoch_metrics = {
            'loss': []
        }
        self.len_inputs = logs['len_inputs']
        self.has_val_data = logs['has_val_data']
        self.has_metrics = logs['has_metrics']
        if self.has_val_data:
            self.epoch_metrics['val_loss'] = []
        if self.has_metrics:
            self.metrics = []
            for metric in logs['metrics'].metrics:
                self.epoch_metrics.update({metric._name: []})
                self.metrics.append(metric._name)
                if self.has_val_data:
                    self.epoch_metrics.update({"val_"+metric._name: []})

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_metrics = {
            'loss': 0.
        }
        if self.has_metrics:
            for metric in self.metrics:
                self.batch_metrics.update({metric: 0.})
                if self.has_val_data:
                    self.batch_metrics.update({"val_"+metric: 0.})
        self.samples_seen = 0.

    def on_epoch_end(self, epoch, logs=None):
        for k in self.batch_metrics:
            if 'val_' in k:
                self.epoch_metrics[k].append(self.batch_metrics[k])
            else:
                self.epoch_metrics[k].append(self.batch_metrics[k] /
                                             self.len_inputs)

    def on_batch_end(self, batch, logs=None):
        for k in self.batch_metrics:
            if 'val_' not in k:
                self.batch_metrics[k] += logs[k] * logs['batch_size']
                self.samples_seen += logs['batch_size']

    def __getitem__(self, name):
        return self.epoch_metrics[name]

    def __repr__(self):
        return str(self.epoch_metrics)

    def __str__(self):
        return str(self.epoch_metrics)


class ModelCheckpoint(Callback):
    """
    Model Checkpoint to save model weights during training

    save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }
    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        th.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    """

    def __init__(self,
                 directory,
                 filename='ckpt.pth.tar',
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=True,
                 verbose=0):
        """
        Model Checkpoint to save model weights during training

        Arguments
        ---------
        file : string
            file to which model will be saved.
            It can be written 'filename_{epoch}_{loss}' and those
            values will be filled in before saving.
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        save_best_only : boolean
            whether to only save if monitored value has improved
        save_weight_only : boolean
            whether to save entire model or just weights
            NOTE: only `True` is supported at the moment
        verbose : integer in {0, 1}
            verbosity
        """
        if directory.startswith('~'):
            directory = os.path.expanduser(directory)
        self.directory = directory
        self.filename = filename
        self.file = os.path.join(self.directory, self.filename + '.pth.tar')
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        # mode = 'min' only supported
        self.best_loss = math.inf
        super(ModelCheckpoint, self).__init__()

    def save_checkpoint(self, epoch, is_best=False):
        info_dict = {
            'epoch': epoch + 1,
            'state_dict': self.trainer.model.state_dict(),
            'optimizer': self.trainer._optimizer.state_dict(),
            'loss': self.trainer.history['loss'][-1],
            'val_loss': self.trainer.history['val_loss'][-1],
        }

        th.save(info_dict, self.file)
        if is_best:
            shutil.copyfile(self.file,
                            os.path.join(self.directory,
                                         '{}_model_best.pth.tar'.
                                         format(self.filename))
                            )

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        is_best = current_loss < self.best_loss

        if self.save_best_only:
            if is_best:
                if self.verbose > 0:
                    print('\nEpoch %i: improved from %0.4f to %0.4f \
                            saving model to %s' % (epoch+1, self.best_loss,
                                                   current_loss, self.file))
                self.best_loss = current_loss
                self.save_checkpoint(epoch, is_best)
        else:
            if self.verbose > 0:
                print('\nEpoch %i: saving model to %s' % (epoch+1, self.file))
            if is_best:
                self.best_loss = current_loss

            self.save_checkpoint(epoch, is_best)


# Estudar
class EarlyStopping(Callback):
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs

        Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0
        super(EarlyStopping, self).__init__()

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.trainer._stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping \
                  at Epoch %04i' % (self.stopped_epoch))


class CSVLogger(Callback):
    """
    Logs epoch-level metrics to a CSV file
    """

    def __init__(self,
                 file,
                 separator=',',
                 append=False):
        """
        Logs epoch-level metrics to a CSV file

        Arguments
        ---------
        file : string
            path to csv file
        separator : string
            delimiter for file
        apped : boolean
            whether to append result to existing file or make new file
        """
        self.file = file
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        self.start_time = logs['start_time']
        if self.append:
            if os.path.exists(self.file):
                with open(self.file) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.file, 'a')
        else:
            self.csv_file = open(self.file, 'w')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        RK = {'num_batches', 'num_epoch', 'batch_size'}

        epc = {key: value[epoch] for key, value in
               self.trainer.history.epoch_metrics.items()}

        def handle_value(k):
            is_zero_dim_tensor = isinstance(k, th.Tensor) and k.dim() == 0
            if isinstance(k, Iterable) and not is_zero_dim_tensor:
                if not isinstance(k, list):
                    k = [k]
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(epc.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] +
                                         [k for k in self.keys
                                          if k not in RK],
                                         dialect=CustomDialect
                                         )
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(epc[key])) for key in
                        self.keys if key not in RK)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.stop_time = logs['stop_time']
        delta = self.stop_time - self.start_time
        print("It took " + str(delta.days) + " days " +
              str(delta.seconds) + " seconds and " +
              str(delta.microseconds) + " microseconds.")
        time = str(delta.days) + " days " + \
            str(delta.seconds) + "s " + \
            str(delta.microseconds) + "mcs."
        row_dict = OrderedDict({'epoch': time})
        self.writer.writerow(row_dict)
        self.csv_file.close()
        self.writer = None


class LambdaCallback(Callback):
    """
    Callback for creating simple, custom callbacks on-the-fly.
    """
    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None,
                 **kwargs):
        super(LambdaCallback, self).__init__()
        self.__dict__.update(kwargs)
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda epoch, logs: None
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda epoch, logs: None
        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda batch, logs: None
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda batch, logs: None
        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None
        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None


class LearningRateFinder(Callback):
    #     https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
    def __init__(self,
                 trainer,
                 stop_factor=4,
                 beta=0.98
                 ):
        self.trainer = trainer
        self.best_loss = 1e9
        self.stop_factor = stop_factor
        self.beta = beta
        # initialize our list of learning rates and losses,
        # respectively
        self.lrs = []
        self.losses = []
        # initialize our learning rate multiplier, average loss, best
        # loss found thus far, current batch number, and weights file
        self.lr_mult = 1
        self.avg_loss = 0
        self.batch_num = 0
        self.weights_file = None

    def reset(self):
        # re-initialize all variables from our constructor
        self.lrs = []
        self.losses = []
        self.lr_mult = 1
        self.avg_loss = 0
        self.best_loss = 1e9
        self.batch_num = 0
        self.weights_file = None

    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = self.trainer._optimizer.param_groups[0]['lr']
        self.lrs.append(lr)
        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value
        cur_loss = logs["loss"]
        self.batch_num += 1
        self.avg_loss = (self.beta * self.avg_loss) + \
                        ((1 - self.beta) * cur_loss)
        smooth = self.avg_loss / (1 - (self.beta ** self.batch_num))
        self.losses.append(smooth)
        # compute the maximum loss stopping factor value
        stop_loss = self.stop_factor * self.best_loss
        # check to see whether the loss has grown too large
        if self.batch_num > 1 and smooth > stop_loss:
            # stop returning and return from the method
            self.trainer._stop_training = True
            return
        # check to see if the best loss should be updated
        if self.batch_num == 1 or smooth < self.best_loss:
            self.best_loss = smooth
        # increase the learning rate
        lr *= self.lr_mult
        for pg in self.trainer._optimizer.param_groups:
            pg['lr'] = lr

    def find(self,
             data,
             start_LR,
             end_LR,
             batch_size,
             loss=None,
             optimizer='adam',
             sample_size=2048,
             epochs=None,
             cuda_device=True):

        # reset our class-specific variables
        self.reset()
        print("#"*10)
        if loss is None:
            loss = th.nn.L1Loss(reduction='mean')

        num_samples = len(data.train.inputs[0])
        steps_per_epoch = np.ceil(num_samples / float(batch_size))

        if epochs is None:
            epochs = int(np.ceil(sample_size / float(steps_per_epoch)))

        print("Finding LR in {} epochs and {} steps".format(epochs,
                                                            steps_per_epoch))

        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        num_batch_updates = epochs * steps_per_epoch
        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lr_mult = (end_LR / start_LR) ** (1.0 / num_batch_updates)

        # create a temporary file path for the model weights and
        # then save the weights (so we can reset the weights when we
        # are done)
        # self.weightsFile = tempfile.mkstemp()[1]
        # self.model.save_weights(self.weightsFile)

        callback = LambdaCallback(on_batch_end=lambda batch, logs:
                                  self.on_batch_end(batch, logs))
        start_time = _get_current_time()
        self.trainer.compile(loss=loss,
                             optimizer=optimizer,
                             callbacks=callback)

        # grab the *original* learning rate (so we can reset it
        # later), and then set the *starting* learning rate
        origLR = self.trainer._optimizer.param_groups[0]['lr']
        for pg in self.trainer._optimizer.param_groups:
            pg['lr'] = start_LR

        self.trainer.fit_loader(data.train_dataloader(),
                                data.val_dataloader(),
                                num_epoch=epochs,
                                cuda_device=cuda_device)

        stop_time = _get_current_time()
        delta = stop_time - start_time
        print("It took " + str(delta.days) + " days " +
              str(delta.seconds) + " seconds and " +
              str(delta.microseconds) + " microseconds.")

        for pg in self.trainer._optimizer.param_groups:
            pg['lr'] = origLR

    def save_csv(self, file_name="lr_loss.csv"):
        axes = np.transpose([self.lrs, self.losses])
        df = pd.DataFrame(axes)
        df.columns = ["lr", "loss"]
        df.to_csv(file_name, index=False)


class CyclicLR(Callback):
    #     https://github.com/bckenstler/CLR
    def __init__(self, base_lr=0.001, max_lr=0.006,
                 step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None,
                 scale_mode='cycle'):

        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        # initialize our list of learning rates and iterations,
        # respectively
        self.lrs = []
        self.iterations = []

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):

        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        self.lrs = []
        self.iterations = []
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr) * \
                   np.maximum(0, (1-x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr) * \
                   np.maximum(0, (1-x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            for pg in self.trainer._optimizer.param_groups:
                pg['lr'] = self.base_lr
        else:
            for pg in self.trainer._optimizer.param_groups:
                pg['lr'] = self.clr()

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        lr = self.trainer._optimizer.param_groups[0]['lr']

        self.lrs.append(lr)
        self.iterations.append(self.trn_iterations)

        for pg in self.trainer._optimizer.param_groups:
            pg['lr'] = self.clr()


def path_to_string(path):
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def convert_img_plot(img):
    img = rescale_intensity(img.cpu().detach().numpy(), out_range=(0, 255))
    im = Image.fromarray(np.float32(img)).convert('LA')
    pil_to_tensor = transforms.ToTensor()(im)
    return pil_to_tensor


class TensorBoardCB(Callback):

    def __init__(self,
                 log_dir='runs',
                 max_img_grid=12):
        self.max_img_grid = max_img_grid
        self.log_dir = log_dir

    def on_train_begin(self, logs={}):
        self._epoch = 0
        self.has_val_data = logs['has_val_data']
        batch_size = logs['len_inputs']/logs['num_batches']
        self._writer = SummaryWriter(comment=path_to_string(
                                                self.log_dir))

        self.max_img_grid = self.max_img_grid if \
            self.max_img_grid < batch_size else batch_size

    def on_batch_end(self, batch, logs=None):
        pil_images = []
        for i in range(self.max_img_grid):
            img = self.batch_imgs[i, 0, :, :, 40]
            pil_images.append(convert_img_plot(img))
        grid = torchvision.utils.make_grid(pil_images)
        self._writer.add_image('images'+str(batch+1), grid, 0)

    def on_epoch_end(self, epoch, logs=None):
        metrics = []
        logs = {key: value[epoch] for key, value in
                self.trainer.history.epoch_metrics.items()}
        for key in logs.keys():
            if 'val' not in key and "_metric" in key:
                if key not in metrics:
                    metrics.append(key)

        for metric in metrics:
            name = metric.split("_metric")[0]
            self._writer.add_scalar(
                name.upper()+'/Train', logs[metric], epoch)
            if self.has_val_data:
                self._writer.add_scalar(name.upper()+'/Val',
                                        logs['val_' + metric], epoch)

        self._writer.add_scalar("Loss/Train", logs['loss'], epoch)
        if self.has_val_data:
            self._writer.add_scalar("Loss/Val", logs['val_loss'], epoch)

    def on_train_end(self, logs=None):
        self._writer.close()
