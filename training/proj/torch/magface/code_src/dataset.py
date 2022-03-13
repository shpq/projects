from proj.torch.magface.code_src.augmentation import Augmentation
from utils import open_image
import random
import torch
import os


class Dataset(torch.utils.data.Dataset):
    """
    Magface dataset class
    """

    def __init__(self, cfg, mode="train"):
        self.cfg = cfg

        dataset_path = cfg.project.dataset.path
        ids_paths = os.listdir(dataset_path)

        limit_faces = cfg.project.dataset.limit_faces
        test_size = cfg.project.training.test_size

        ids2imgs = {
            i: os.listdir(os.path.join(dataset_path, i)) for i in ids_paths
        }
        ids2imgs = {k: v for k, v in ids2imgs.items() if self.selected(v)}
        for i, imgs in ids2imgs.items():
            ids2imgs[i] = self.sample(imgs, mode, limit_faces, test_size)

        self.pairs = [
            (i, os.path.join(dataset_path, pth, img))
            for i, (pth, imgs) in enumerate(ids2imgs.items())
            for img in imgs
        ]
        self.offset = self.pairs[-1][0]
        random.shuffle(self.pairs)
        self.transform = Augmentation(cfg, mode=mode).augment

    def sample(self, imgs, mode, limit_faces, test_size):
        imgs = [img for img in imgs if img.endswith(".jpg")]
        N_sample = min(len(imgs), limit_faces)
        random.seed(220)
        imgs = random.sample(imgs, N_sample)
        N_test = int(test_size * N_sample)
        if mode == "test":
            return imgs[:N_test]
        return imgs[N_test:]

    def selected(self, v):
        min_faces = self.cfg.project.dataset.min_faces
        max_faces = self.cfg.project.dataset.max_faces
        return min_faces <= len(v) <= max_faces

    def __getitem__(self, index):
        idx, path = self.pairs[index]
        image = open_image(path)
        return self.transform(image), torch.tensor(idx, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)
