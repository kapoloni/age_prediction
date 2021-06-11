"""
DataSet Module
"""
# Standard library imports
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import torch
import random

# Third party imports
import SimpleITK as sitk
from torch.utils import data
import torchio as tio
import torchio.transforms as transforms

# Local application imports
from .utils import (_process_array_argument)


class MyDataSet(data.Dataset):
    """
    DataSet created for this specific application
    """

    def __init__(self,
                 inputs: list,
                 targets: list = [],
                 lims_intensity: list = [0, 100],
                 input_transform=None,
                 ):
        self.inputs = _process_array_argument(inputs)
        self.targets = _process_array_argument(targets)
        self.input_transform = input_transform
        self.lims_intensity = lims_intensity
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.num_inputs = len(self.inputs)
        self.num_targets = len(self.targets)
        if input_transform is not None:
            self.num_transform = len(self.input_transform)
        else:
            self.num_transform = 1
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
        if self.num_transform > 1:
            x = get_transform(self.lims_intensity,
                              self.input_transform[index])(x)
        else:
            x = get_transform(self.lims_intensity,
                              None)(x)

        # Typecasting
        x = torch.from_numpy(x).type(self.inputs_dtype)
        if self.num_targets > 0:
            y = np.array(self.targets[0][index])
            y = torch.from_numpy(y).type(self.targets_dtype)
            return x, y.unsqueeze(-1)
        else:
            return x


def get_transform(lims_intensity, transformation=None):
    nac_min, nac_max = lims_intensity
    if transformation is not None:
        dict_transform = {
            'bf': tio.RandomBiasField(coefficients=(-0.3, 0.3),
                                      order=3),
            'noise': tio.RandomNoise(std=(5, 30)),
            'trx': translation(0, 10),  # Translation x
            'rtx': translation(2, 10),  # Translation y
            'try': translation(4, 10),  # Translation z
            'rty': rotation(0, 20),
            'trz': rotation(2, 20),
            'rtz': rotation(4, 20),
            'noise+bf':
            tio.Compose([tio.RandomNoise(std=(5, 30)),  # Gaussian
                         # and bias field
                         tio.RandomBiasField(coefficients=(-0.3, 0.3),
                                             order=3)]),
            'rtx+trz':
            tio.Compose([rotation(0, 15),  # Small rotation x
                        translation(4, 8)]),  # and translation z,
            'rtx+try':
            tio.Compose([rotation(0, 15),  # Small rotation x
                        translation(2, 8)]),  # and translation y,
            'rtx+trx':
            tio.Compose([rotation(0, 15),  # Small rotation x
                        translation(0, 8)]),  # and translation x
            'rty+trz':
            tio.Compose([rotation(2, 15),  # Small rotation y
                        translation(4, 8)]),  # and translation z
            'rty+try':
            tio.Compose([rotation(2, 15),  # Small rotation y
                        translation(2, 8)]),  # and translation y
            'rty+trx':
            tio.Compose([rotation(2, 15),  # Small rotation y
                        translation(0, 8)]),  # and translation x
            'rtz+trz':
            tio.Compose([rotation(4, 15),  # Small rotation x
                        translation(4, 8)]),  # and translation z
            'rtz+try':
            tio.Compose([rotation(4, 15),  # Small rotation z
                        translation(2, 8)]),  # and translation y
            'rtz+trx':
            tio.Compose([rotation(4, 15),  # Small rotation z
                        translation(0, 8)]),  # and translation 0
            'rtz+trx+bf':
            tio.Compose([rotation(4, 15),  # Small rotation z
                         translation(0, 8),  # and translation x
                         # and bias field
                         tio.RandomBiasField(coefficients=(-0.3, 0.3),
                                             order=3)]),
            'rtz+noise+bf':
            tio.Compose([rotation(0, 15),  # Small rotation z
                         tio.RandomNoise(std=(5, 30)),  # Gaussian
                         # and bias field
                         tio.RandomBiasField(coefficients=(-0.3, 0.3),
                                             order=3)])
            }
        return tio.Compose([
                    transforms.RescaleIntensity(out_min_max=(
                                                nac_min, nac_max),
                                                percentiles=(0.5, 99.5)),
                    dict_transform[transformation]
                    ])
    else:
        return tio.Compose([
                    transforms.RescaleIntensity(out_min_max=(
                                                nac_min, nac_max),
                                                percentiles=(0.5, 99.5)),
                    ])


def translation(axis, th):
    transl = np.zeros((6))
    transl[axis] = -th
    transl[axis+1] = th

    return tio.RandomAffine(scales=(1, 1, 1, 1, 1, 1),
                            translation=tuple(transl),
                            degrees=(0, 0, 0, 0, 0, 0),
                            image_interpolation='bspline',
                            center='image',
                            default_pad_value=0)


def rotation(axis, angle):
    angles = np.zeros((6))
    angles[axis] = -angle
    angles[axis+1] = angle

    return tio.RandomAffine(degrees=tuple(angles),
                            scales=(1, 1, 1, 1, 1, 1),
                            image_interpolation='bspline',
                            center='image',
                            default_pad_value=0)


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
        adni, label_adni = self.get_imgs_label(os.path.join(self.database,
                                                            "ADNI"))
        ixi, label_ixi = self.get_imgs_label(os.path.join(self.database,
                                                          'IXI'))

        self.scan_paths = np.concatenate([adni, ixi])
        self.transform = [None for _ in range(len(self.scan_paths))]
        label_train = dict(label_adni)
        label_train.update(label_ixi)

        # DataAugmented
        if self.data_aug:
            aug_scans, aug_transf, aug_labels = self.generate_dataAugmentation(
                                                self.scan_paths, label_train)
            self.scan_paths = np.concatenate([self.scan_paths,
                                              aug_scans])
            label_train.update(aug_labels)
            self.transform = np.concatenate([self.transform,
                                             aug_transf])

        self.y = np.array([label_train[tr.split("/")[-1]]
                           for tr in self.scan_paths])

        # data = pd.concat([pd.DataFrame(self.scan_paths),
        #                   pd.DataFrame(self.transform),
        #                   pd.DataFrame(self.y)], axis=1)
        # data.columns = ['Path', 'Transform', 'Label']
        # data = data.sample(frac=1).reset_index(drop=True)
        # return [data.Path, data.Transform, data.Label]
        return [self.scan_paths, self.transform, self.y]

    def generate_dataAugmentation(self, scan_paths, y):
        y = np.array([y[tr.split("/")[-1]]
                      for tr in scan_paths])

        dt = pd.concat([pd.DataFrame(scan_paths), pd.DataFrame(y)],
                       axis=1).reset_index(drop=True)
        dt.columns = ['Image', 'Age']

        if self.age_range[1] == 70:
            th = 300
        else:
            th = 200
        dict_transf = ['bf', 'noise', 'trx', 'rtx', 'try',
                       'rty', 'trz', 'rtz', 'noise+bf',
                       'rtx+trz', 'rtx+try', 'rtx+trx',
                       'rty+trz', 'rty+try', 'rty+trx',
                       'rtz+trx', 'rtz+try', 'rtz+trz',
                       'rtz+trx+bf', 'rtz+noise+bf']

        delta = dt['Age'].max() - dt['Age'].min()
        discrete = pd.cut(dt['Age'], bins=int(delta/3),  # three bins
                          right=False).reset_index(drop=True).astype(str)
        dt['Age_disc'] = discrete.reset_index(drop=True)
        aug_dt = pd.DataFrame()
        for age in dt['Age_disc'].value_counts().keys():
            age_dt = dt[dt['Age_disc'] == age]
            total = len(age_dt['Age_disc'])
            mult = np.ceil((th-total) / total)
            new_dt = pd.concat([age_dt] * int(mult))
            transfs = dict_transf * \
                int(np.ceil(len(new_dt)/len(dict_transf)))
            random.shuffle(transfs)
            new_dt['transform'] = transfs[: len(new_dt)]
            aug_dt = pd.concat([aug_dt, new_dt])
        scan_paths = aug_dt.Image.values
        transf = aug_dt['transform'].values
        y = dict([(img.split("/")[-1], y)
                  for img, y in zip(aug_dt.Image, aug_dt.Age)])

        return scan_paths, transf, y

    def load_eval_data(self):
        # val paths
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
        adni, _ = self.get_imgs_label(os.path.join(self.database, 'ADNI'))
        ixi, _ = self.get_imgs_label(os.path.join(self.database, 'IXI'))

        self.scan_paths = np.concatenate([adni, ixi])

        return [self.scan_paths]
