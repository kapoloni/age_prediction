import os
import torch
import numpy as np
import torchio as tio
import SimpleITK as sitk
from torch.utils import data

# Local import
from age_prediction.pandas_reader import PandasReader
from age_prediction.utils import (_process_array_argument)


class MyDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list = [],
                 input_transform=None
                 ):
        self.inputs = _process_array_argument(inputs)
        self.targets = _process_array_argument(targets)
        self.input_transform = input_transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.num_inputs = len(self.inputs)
        self.num_targets = len(self.targets)
        self.info = []

    def __len__(self):
        return len(self.inputs) if not \
               isinstance(self.inputs, (tuple, list)) else \
               len(self.inputs[0])

    def __getitem__(self, index):
        input_ID = self.inputs[0][index]
        self.info = [input_ID, index]

        # Read input image
        x = sitk.GetArrayFromImage(tio.ScalarImage(input_ID).as_sitk())[
            np.newaxis, :, :, :]

        # Preprocessing
        if self.input_transform is not None:
            x = self.input_transform(x)

        # Typecasting
        x = torch.from_numpy(x).type(self.inputs_dtype)
        if self.num_targets > 0:
            y = np.array(self.targets[0][index])
            y = torch.from_numpy(y).type(self.targets_dtype)
            return x, y.unsqueeze(-1)
        else:
            return x


class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inp):
        for t in self.transforms:
            inp = t(inp)
        return inp

    def __repr__(self):
        return str([transform for transform in self.transforms])


class LoadDataPath(object):

    def __init__(self,
                 database: str,
                 csv_data: str,
                 side: str,
                 train: bool,
                 val: bool,
                 data_aug: True,
                 age_range=None,
                 train_file='train_all.csv',
                 val_file='val_exp.csv',
                 test_file='test_exp.csv'
                 ):
        self.database = database
        self.csv_data = csv_data
        self.side = side
        self.data_aug = data_aug
        self.train = train
        self.val = val
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        if age_range is not None:
            if not isinstance(age_range, list):
                raise ValueError('Age_range should be a list')
        self.age_range = age_range

        if self.train:
            self.load_train_data()
        elif self.val:
            self.load_eval_data()
        else:
            self.load_test_data()

    def load_train_data(self):
        # train paths
        adni, ixi, label_adni, label_ixi = [], [], [], []
        adni, label_adni = PandasReader(folder=os.path.join(self.database,
                                                            "ADNI"),
                                        side=self.side,
                                        infos=os.path.join(self.csv_data,
                                                           self.train_file),
                                        age_range=self.age_range
                                        ).get_imgs_label()
        ixi, label_ixi = PandasReader(folder=os.path.join(self.database,
                                                          'IXI'),
                                      side=self.side,
                                      infos=os.path.join(self.csv_data,
                                                         self.train_file),
                                      age_range=self.age_range
                                      ).get_imgs_label()

        self.scan_paths = np.concatenate([adni, ixi])
        label_train = dict(label_adni)
        label_train.update(label_ixi)

        # DataAugmented
        if self.data_aug:
            adni_ag, ixi_ag, label_adni_ag, label_ixi_ag = [], [], [], []
            adni_ag, label_adni_ag = PandasReader(folder=os.path.join(
                                                        self.database,
                                                        'dataAug',
                                                        'ADNI'),
                                                  side=self.side,
                                                  infos=os.path.join(
                                                      self.csv_data,
                                                      self.train_file),
                                                  age_range=self.age_range
                                                  ).get_imgs_label()
            ixi_ag, label_ixi_ag = PandasReader(folder=os.path.join(
                                                      self.database,
                                                      'dataAug',
                                                      'IXI'),
                                                side=self.side,
                                                infos=os.path.join(
                                                    self.csv_data,
                                                    self.train_file),
                                                age_range=self.age_range
                                                ).get_imgs_label()

            self.scan_paths = np.concatenate([self.scan_paths,
                                              adni_ag,
                                              ixi_ag])

            label_train.update(label_adni_ag)
            label_train.update(label_ixi_ag)

        self.y = np.array([label_train[tr.split("/")[-1]]
                           for tr in self.scan_paths])

        return [self.scan_paths, self.y]

    def load_eval_data(self):
        # val paths
        adni, ixi, label_adni, label_ixi = [], [], [], []
        adni, label_adni = PandasReader(folder=os.path.join(self.database,
                                                            'ADNI'),
                                        side=self.side,
                                        infos=os.path.join(self.csv_data,
                                                           self.val_file),
                                        ).get_imgs_label()
        ixi, label_ixi = PandasReader(folder=os.path.join(self.database,
                                                          'IXI'),
                                      side=self.side,
                                      infos=os.path.join(self.csv_data,
                                                         self.val_file),
                                      ).get_imgs_label()

        self.scan_paths = np.concatenate([adni, ixi])
        label_val = dict(label_adni)
        label_val.update(label_ixi)

        self.y = np.array([label_val[tr.split("/")[-1]]
                           for tr in self.scan_paths])

        return [self.scan_paths, self.y]

    def load_test_data(self):
        # test paths
        adni, ixi, label_adni, label_ixi = [], [], [], []
        adni, label_adni = PandasReader(folder=os.path.join(self.database,
                                                            'ADNI'),
                                        side=self.side,
                                        infos=os.path.join(self.csv_data,
                                                           self.val_file),
                                        ).get_imgs_label()
        ixi, label_ixi = PandasReader(folder=os.path.join(self.database,
                                                          'IXI'),
                                      side=self.side,
                                      infos=os.path.join(self.csv_data,
                                                         self.val_file),
                                      ).get_imgs_label()

        self.scan_paths = np.concatenate([adni, ixi])

        return [self.scan_paths]
