from proj.tensorflow.human_segmentation.code_src.augmentation import (
    Augmentation,
)
from random import shuffle, sample, seed, random, choice
from utils import open_image
from PIL import Image
import numpy as np
import json
import cv2
import os


class DataGenerator:
    def __init__(self, cfg, mode="train"):
        seed(166)
        dataset_path = cfg.project.dataset.path
        self.background_path = cfg.project.dataset.backgrounds_path
        image_names = os.listdir(dataset_path)
        self.cfg = cfg
        self.size = cfg.project.training.size
        self.mode = mode
        self.transform = Augmentation(cfg, mode).augment
        self.batch_size = cfg.project.training.batch_size
        image_list_path = cfg.project.dataset.image_list_path
        if image_list_path is None:
            self.image_paths = [
                os.path.join(dataset_path, n)
                for n in image_names
                if self.is_image(n)
            ]
        else:
            self.image_paths = json.load(open(image_list_path, "r"))
            self.image_paths = [
                os.path.join(dataset_path, n) for n in self.image_paths
            ]

        test_size = cfg.project.training.test_size
        test_N = int(len(self.image_paths) * test_size)
        sampled = sample(self.image_paths, test_N)
        if self.mode == "test":
            self.image_paths = sampled
        else:
            self.image_paths = list(set(self.image_paths).difference(sampled))

        seed(166)
        self.backgrounds = sample(os.listdir(self.background_path), 60000)
        self.on_epoch_end()

    def is_image(self, name):
        return (
            name.endswith(".png")
            and not name.endswith("mask.png")
            and not name.endswith("mask_paddle.png")
        )

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        return self.__get_data(index)

    def on_epoch_end(self):
        if self.cfg.project.training.shuffle:
            shuffle(self.image_paths)

    def change_background(self, image, mask):
        mask = np.array(mask) / 255
        random_background = choice(self.backgrounds)
        background_path = os.path.join(self.background_path, random_background)
        background = open_image(background_path)
        background = np.array(background.resize(image.size))
        mask = np.expand_dims(mask, -1)
        # get human alpha composition image
        image = np.array(image) * mask
        # add background
        image += background * (1 - mask)
        return Image.fromarray(image.astype("uint8"))

    def mask2trimap(self, mask, eroision_iter=12, dilate_iter=12):
        mask = (mask * 255).astype("uint8")
        d_kernel = np.ones((3, 3))
        erode = cv2.erode(mask, d_kernel, iterations=eroision_iter).astype(
            "float"
        )
        dilate = cv2.dilate(mask, d_kernel, iterations=dilate_iter).astype(
            "float"
        )
        erode /= 2
        dilate /= 2
        trimap = erode + dilate
        return trimap / 255

    def __get_data(self, index):
        X = np.empty((self.batch_size, *self.size, 3))
        y = np.zeros((self.batch_size, *self.size))
        trimap = np.zeros((self.batch_size, *self.size))

        for i, id in enumerate(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ):
            image_path = self.image_paths[id]
            image_raw = open_image(image_path)
            image_mask = open_image(image_path + "mask_paddle.png", RGB=False)
            image_trimap = open_image(image_path + "trimap.png", RGB=False)
            if self.mode == "train":
                if self.cfg.project.training.use_random_background:
                    if random() > self.cfg.project.training.prob_background:
                        image_raw = self.change_background(
                            image_raw, image_mask
                        )
            aug_image, aug_mask, aug_trimap = self.transform(
                image_raw, image_mask, image_trimap
            )

            X[
                i,
            ] = aug_image
            y[
                i,
            ] = aug_mask
            trimap[
                i,
            ] = aug_trimap

        return X, y, trimap
