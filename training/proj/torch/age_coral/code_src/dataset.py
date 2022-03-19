from proj.torch.age_coral.code_src.augmentation import Augmentation
from utils import open_image
from random import sample, seed
import torch
import os


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class
    """

    def __init__(self, cfg, mode="train"):
        dataset_path = cfg.project.dataset.path
        age_dirs = [x for x in os.listdir(dataset_path) if x.isnumeric()]
        self.mode = mode
        self.num_classes = cfg.project.dataset.num_classes
        self.image_paths = [os.path.join(dataset_path, age, img)
                            for age in age_dirs
                            for img in os.listdir(os.path.join(
                                dataset_path, age))]

        test_size = cfg.project.training.test_size
        test_len = int(len(self.image_paths) * test_size)
        seed(166)
        sampled = sample(self.image_paths, test_len)
        if self.mode == "test":
            self.image_paths = sampled
        else:
            self.image_paths = list(set(self.image_paths).difference(sampled))

        self.image_pairs = [(int(p.split(os.sep)[-2]), p)
                            for p in self.image_paths]
        self.transform = Augmentation(cfg, mode=mode).augment

    def __getitem__(self, index):
        target, path = self.image_pairs[index]
        image = open_image(path)
        levels = [1] * target + [0] * (self.num_classes - target)
        levels = torch.tensor(levels, dtype=torch.float32)
        return self.transform(image), target, levels

    def __len__(self):
        return len(self.image_pairs)
