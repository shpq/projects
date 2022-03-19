import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class Augmentation:
    """
    Augmentation class
    """

    def __init__(self, cfg, mode="train"):
        self.mode = mode
        self.size = cfg.project.training.size
        self.p = cfg.project.training.p_augs

    def augment_train(self, image):
        compose_list = [
            A.Resize(*self.size),
            A.ShiftScaleRotate(
                shift_limit=0.01, scale_limit=0.01, rotate_limit=5, p=1
            ),
            A.Downscale(p=self.p),
            A.CLAHE(p=self.p),
            A.RandomContrast(p=self.p),
            A.RandomBrightness(p=self.p),
            A.HorizontalFlip(),
            A.Blur(blur_limit=2, p=self.p),
            A.GridDistortion(p=1, num_steps=12, distort_limit=0.7),
            A.HueSaturationValue(p=self.p),
            A.ToGray(p=self.p),
            A.JpegCompression(p=self.p),
            A.Cutout(p=self.p),
            A.GaussNoise(var_limit=(0, 50), p=self.p),
            A.ToGray(p=self.p),
            A.Normalize(),
            ToTensorV2(),
        ]
        comp = A.Compose(compose_list, p=1)
        return comp(image=image)["image"]

    def augment_test(self, image):
        compose_list = [A.Resize(*self.size), A.Normalize(), ToTensorV2()]
        comp = A.Compose(compose_list, p=1)
        return comp(image=image)["image"]

    def augment(self, image):
        image = np.array(image).astype("uint8")
        if self.mode == "train":
            return self.augment_train(image)
        elif self.mode == "test":
            return self.augment_test(image)
        else:
            message = "Augment mode should be in (train, test)"
            message += f"but equals {self.mode}"
            raise ValueError(message)
