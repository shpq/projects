import os
import random
import numpy as np
from random import shuffle
from utils import open_image
from proj.frmwrk.example_project.code_src.augmentation import Augmentation


class DataGenerator:
    def __init__(self, cfg, mode="train"):
        random.seed(166)
        dataset_path = cfg.project.dataset.path
        image_names = os.listdir(dataset_path)
        self.cfg = cfg
        self.size = cfg.project.training.size
        self.mode = mode
        self.transform = Augmentation(cfg, mode).augment
        self.batch_size = cfg.project.training.batch_size
        self.image_paths = [os.path.join(dataset_path, n) for n in image_names]

        # test_size = cfg.project.training.test_size
        # split images for train and test without intersection

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        X, y = self.__get_data(index)
        return X, y

    def on_epoch_end(self):
        if self.cfg.training.shuffle:
            shuffle(self.image_paths)

    def __get_data(self, index):
        X = np.empty((self.batch_size, *self.size, 3))
        y = np.zeros((self.batch_size, self.cfg.project.dataset.num_classes))

        for i, id in enumerate(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ):
            image_raw = open_image(self.image_paths[id])

            # get target
            target = 0
            augmented_image = self.transform(image_raw)
            X[
                i,
            ] = augmented_image
            y[i, target] = 1

        return X, y
