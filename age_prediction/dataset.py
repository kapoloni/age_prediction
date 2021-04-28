"""
DataSet Module
"""
# Standard library imports
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import torch

# Third party imports
import SimpleITK as sitk
import torchio as tio
from torch.utils import data

# Local application imports
from .utils import (_process_array_argument)


class MyDataSet(data.Dataset):
    """
    DataSet created for this specific application
    """

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
    """
        Dataset class for loading in-memory data.

        Arguments
        ---------
        inputs: (numpy array or list)
            Input image paths
        targets : (numpy array or list)
            Targets
        input_transform : transform functions
            transform to apply to input sample individually
    """
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
    """
    Class to load the image paths.
    Created for this specific application.
    """

    def __init__(self,
                 database: str,
                 csv_data: str,
                 side: str,
                 stage: str = 'train',
                 data_aug: bool = True,
                 age_range: list = None,
                 train_file: str = 'train_all.csv',
                 val_file: str = 'val_70-100.csv',
                 test_file: str = 'test_70-100.csv'
                 ):
        self.database = database
        self.csv_data = csv_data
        self.side = side
        self.data_aug = data_aug
        self.stage = stage
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        if age_range is not None:
            if not isinstance(age_range, list):
                raise ValueError('Age_range should be a list')
        self.age_range = age_range

        if 'train' in self.stage:
            self.infos = os.path.join(self.csv_data, self.train_file)
            self.load_train_data()
        elif 'val' in self.stage:
            self.infos = os.path.join(self.csv_data, self.val_file)
            self.load_eval_data()
        else:
            self.infos = os.path.join(self.csv_data, self.test_file)
            self.load_test_data()

        """
            Class for reading and format the image name paths and labels

            Arguments
            ---------
            database: (string)
                Database name
            csv_data: (string)
                Folder that contains the csv data
            side : ({L or R}, string)
                Hippocampal side image
            stage: ({train, val, test} string)
                If is train, val or test data
            age_range : (List, optional)
                Age range to delimit the train
                Default: None
            train_file: (string, optional)
                Filename with the image names for training
                Default: train_all.csv
            val_file: (string, optional)
                Filename with the image names for validation
                Default: val_70-100.csv
            test_file: (string, optional)
                Filename with the image names for validation
                Default: test_70-100.csv
        """

    def get_imgs_label(self, folder):
        img_infos = pd.read_csv(self.infos)
        if self.age_range is not None:
            img_infos = img_infos[(img_infos.Age >= self.age_range[0]) &
                                  (img_infos.Age < self.age_range[1])]

        scan_paths = [
            os.path.join(os.getcwd(), folder, x)
            for x in os.listdir(folder)
            if "scale" not in x  # Remove data augmentation of scale
            if self.side in x
            if x.split(self.side)[0] in img_infos['Image Filename'].values
        ]
        labels = self.get_age(scan_paths, img_infos, self.side)

        return scan_paths, labels

    def get_age(self, path, ref, side):
        label = {}
        for file in path:
            img_filename = file.split("/")[-1]
            for age, img_file in ref[['Age', 'Image Filename']].values:
                if img_filename.split(side)[0] in img_file:
                    label[img_filename] = age
        return label

    def load_train_data(self):
        # train paths
        adni, ixi, label_adni, label_ixi = [], [], [], []
        adni, label_adni = self.get_imgs_label(os.path.join(self.database,
                                                            "ADNI"))
        ixi, label_ixi = self.get_imgs_label(os.path.join(self.database,
                                                          'IXI'))

        self.scan_paths = np.concatenate([adni, ixi])
        label_train = dict(label_adni)
        label_train.update(label_ixi)

        # DataAugmented
        if self.data_aug:
            adni_ag, ixi_ag, label_adni_ag, label_ixi_ag = [], [], [], []
            adni_ag, label_adni_ag = self.get_imgs_label(
                                        os.path.join(self.database,
                                                     'dataAug', 'ADNI'))
            ixi_ag, label_ixi_ag = self.get_imgs_label(
                                        os.path.join(self.database,
                                                     'dataAug', 'IXI'))

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
        adni, label_adni = self.get_imgs_label(os.path.join(self.database,
                                                            'ADNI'))
        ixi, label_ixi = self.get_imgs_label(os.path.join(self.database,
                                                          'IXI'))

        self.scan_paths = np.concatenate([adni, ixi])
        label_val = dict(label_adni)
        label_val.update(label_ixi)

        self.y = np.array([label_val[tr.split("/")[-1]]
                           for tr in self.scan_paths])

        return [self.scan_paths, self.y]

    def load_test_data(self):
        # test paths
        adni, ixi, label_adni, label_ixi = [], [], [], []
        adni, label_adni = self.get_imgs_label(os.path.join(self.database,
                                                            'ADNI'))
        ixi, label_ixi = self.get_imgs_label(os.path.join(self.database,
                                                          'IXI'))

        self.scan_paths = np.concatenate([adni, ixi])

        return [self.scan_paths]
