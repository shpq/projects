from random import randint
import albumentations as A
import numpy as np
import cv2


def get_channel_value():
    return randint(0, 255)


def get_rgb():
    return [get_channel_value(), get_channel_value(), get_channel_value()]


class Augmentation:
    """
    Augmentation class
    """

    def __init__(self, cfg, mode="train"):
        self.mode = mode
        self.size = cfg.project.training.size
        self.size_mult = cfg.project.training.size_mult
        self.size_for_resizing = tuple(
            int(self.size_mult * s) for s in self.size
        )
        self.p_small = cfg.project.training.probability_hard_aug

    def get_crop_strat(self, image):
        if self.mode == "test":
            return [
                A.Resize(*self.size),
            ]
        else:
            return [
                A.Resize(*self.size_for_resizing),
                A.RandomCrop(*self.size),
            ]

    def augment_train(self, image, mask, trimap):
        crop_strat = self.get_crop_strat(image)
        compose_list = [
            A.HorizontalFlip(),
            A.ShiftScaleRotate(
                shift_limit=0.25,
                scale_limit=0.25,
                rotate_limit=60,
                border_mode=cv2.BORDER_CONSTANT,
                value=get_rgb(),
            ),
            *crop_strat,
            A.Blur(p=self.p_small, blur_limit=(3, 7)),
            A.JpegCompression(
                p=self.p_small,
                quality_lower=30,
                quality_upper=95,
            ),
            A.GaussNoise(p=self.p_small, var_limit=(1e2, 3e3)),
            A.Downscale(p=self.p_small, scale_min=0.4, scale_max=0.9),
            A.GridDistortion(p=self.p_small, num_steps=6, distort_limit=0.4),
            A.ColorJitter(
                p=self.p_small,
                brightness=0.45,
                contrast=0.4,
                saturation=0.1,
                hue=0.1,
            ),
            A.ToGray(p=self.p_small),
            A.RGBShift(p=self.p_small),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1),
        ]
        comp = A.Compose(
            compose_list, p=1, additional_targets={"trimap": "mask"}
        )
        augmented = comp(image=image, mask=mask, trimap=trimap)
        return augmented["image"], augmented["mask"], augmented["trimap"]

    def augment_test(self, image, mask, trimap):
        crop_strat = self.get_crop_strat(image)
        compose_list = [
            *crop_strat,
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1),
        ]
        comp = A.Compose(
            compose_list, p=1, additional_targets={"trimap": "mask"}
        )
        augmented = comp(image=image, mask=mask, trimap=trimap)
        return augmented["image"], augmented["mask"], augmented["trimap"]

    def augment(self, image, mask, trimap):
        image, trimap = np.array(image), np.array(trimap) / 255.0
        image = image.astype("uint8")
        mask = np.array(mask).astype("float64") / 255.0

        if self.mode == "train":
            return self.augment_train(image, mask, trimap)
        elif self.mode == "test":
            return self.augment_test(image, mask, trimap)
        else:
            message = "Augment mode should be in (train, test) "
            message += f"but equals {self.mode}"
            raise ValueError(message)
