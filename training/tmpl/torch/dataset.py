from proj.frmwrk.example_project.code_src.augmentation import Augmentation
from utils import open_image
import torch
import os


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class
    """

    def __init__(self, cfg, mode="train"):
        dataset_path = cfg.project.dataset.path
        image_names = os.listdir(dataset_path)

        # split images for train and test without intersection

        self.image_paths = [os.path.join(dataset_path, n) for n in image_names]
        self.transform = Augmentation(cfg, mode=mode).augment

        # custom process for target

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = open_image(path)
        # get target
        target = 0
        return self.transform(image), target

    def __len__(self):
        return len(self.image_paths)
