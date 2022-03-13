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

    def augment_train(self, image):
        compose_list = [
            A.Resize(*self.size),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.25, rotate_limit=60
            ),
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
