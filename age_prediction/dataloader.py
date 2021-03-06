"""
DataLoader Module
"""
# Standard library imports
import numpy as np

# Third party imports
import SimpleITK as sitk
import torch.nn as nn
import torchio as tio
from torch.utils.data import DataLoader

# Local application imports
from .dataset import MyDataSet, LoadDataPath


class MyDataLoader(nn.Module):
    """
    DataLoader created for this specific application
    """

    def __init__(self,
                 database: list,
                 csv_data: list,
                 side: str,
                 batch: int,
                 data_aug: bool = True,
                 age_range: list = None,
                 train_file: str = 'train_all.csv',
                 val_file: str = 'val_70-100.csv',
                 test_file: str = 'test_70-100.csv',
                 num_workers: int = 10):
        super().__init__()
        self.database = database
        self.csv_data = csv_data
        self.data_aug = data_aug
        self.side = side
        self.batch = batch
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.num_workers = num_workers
        if age_range is not None:
            if not isinstance(age_range, list):
                raise ValueError('Age_range should be a list')
        self.age_range = age_range

        """
            Class to create the DataLoader

            Arguments
            ---------
            database: (string)
                Database name
            csv_data: (string)
                Folder that contains the csv data
            side : ({L or R}, string)
                Hippocampal side image
            batch: (int)
                Size of the batch
            data_aug: (bool, optional)
                Whether to train with data augmentation or not
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
            num_workers: (int, optional)
                Number of cpu to run DataLoader in parallel
                Default: 10 cpus
        """

    def prepare_data(self, stage: str = None):
        if stage == 'fit' or stage is None:
            train_data = LoadDataPath(database=self.database,
                                      csv_data=self.csv_data,
                                      side=self.side,
                                      data_aug=self.data_aug,
                                      stage='train',
                                      age_range=self.age_range,
                                      train_file=self.train_file
                                      )

            train_label = train_data.y
            self.train_transform = train_data.transform
            train_data = train_data.scan_paths

            val_data = LoadDataPath(database=self.database,
                                    csv_data=self.csv_data,
                                    side=self.side,
                                    data_aug=False,
                                    stage='val',
                                    val_file=self.val_file)

            val_label = val_data.y
            val_data = val_data.scan_paths

            self.trainpath = train_data
            self.trainlabel = train_label
            self.valpath = np.array(val_data)
            self.vallabel = val_label

        if stage == 'test' or stage is None:
            test_data = LoadDataPath(database=self.database,
                                     csv_data=self.csv_data,
                                     side=self.side,
                                     data_aug=False,
                                     stage='test',
                                     test_file=self.test_file)

            self.testpath = test_data.scan_paths

        if stage == 'test_label' or stage is None:
            test_data = LoadDataPath(database=self.database,
                                     csv_data=self.csv_data,
                                     side=self.side,
                                     data_aug=False,
                                     stage='val',
                                     val_file=self.test_file)

            self.testpath = test_data.scan_paths
            self.testlabel = test_data.y

    def setup(self, stage=None):
        nac_min, nac_max = nac_image(self.database, self.side)
        if stage == 'fit' or stage is None:
            self.train = MyDataSet(inputs=self.trainpath,
                                   targets=self.trainlabel,
                                   input_transform=self.train_transform,
                                   lims_intensity=[nac_min, nac_max])

            self.val = MyDataSet(inputs=self.valpath,
                                 targets=self.vallabel,
                                 lims_intensity=[nac_min, nac_max])

        if stage == 'test' or stage is None:
            self.test = MyDataSet(inputs=self.testpath,
                                  lims_intensity=[nac_min, nac_max])
        if stage == 'test_label' or stage is None:
            self.test = MyDataSet(inputs=self.testpath,
                                  targets=self.testlabel,
                                  lims_intensity=[nac_min, nac_max])

    def train_dataloader(self):
        return DataLoader(self.train,
                          shuffle=True,
                          batch_size=self.batch,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val,
                          shuffle=True,
                          batch_size=self.batch,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch,
                          num_workers=self.num_workers)

    def testlabel_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch,
                          num_workers=self.num_workers)


def nac_image(database, side):
    img = sitk.GetArrayFromImage(tio.ScalarImage(
        database + '/template/NAC_T1_RAI' + side +
        '_hippocampus.nii.gz').as_sitk())

    return img.min(), img.max()
