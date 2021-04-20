
import os
import pandas as pd


class PandasReader:
    def __init__(self,
                 folder: str,
                 side: str,
                 infos: str,
                 age_range=None,
                 train_file='train_all.csv',
                 val_file='val_exp.csv'
                 ):
        self.folder = folder
        self.side = side
        self.infos = infos
        self.train_file = train_file,
        self.val_file = val_file

        if age_range is not None:
            if not isinstance(age_range, list):
                raise ValueError('Age_range should be a list')
        self.age_range = age_range

    def get_imgs_label(self):
        img_infos = pd.read_csv(self.infos)
        if self.age_range is not None:
            img_infos = img_infos[(img_infos.Age >= self.age_range[0]) &
                                  (img_infos.Age < self.age_range[1])]

        scan_paths = [
            os.path.join(os.getcwd(), self.folder, x)
            for x in os.listdir(self.folder)
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
