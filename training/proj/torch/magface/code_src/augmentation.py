import albumentations.augmentations.geometric.functional as F
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import random
import cv2


class AddBorder(A.core.transforms_interface.DualTransform):
    """
    Add border to image to simulate situation where faces
    are located near corner
    """

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(AddBorder, self).__init__(always_apply, p)
        self.limit = A.core.transforms_interface.to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        img = F.rotate(img, angle, interpolation, self.border_mode, self.value)
        return F.rotate(
            img, -angle, interpolation, self.border_mode, self.value
        )

    def apply_to_mask(self, img, angle=0, **params):
        img = F.rotate(
            img, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value
        )
        return F.rotate(
            img, -angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value
        )

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        return F.bbox_rotate(bbox, angle, params["rows"], params["cols"])

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return F.keypoint_rotate(keypoint, angle, **params)

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")


class Augmentation:
    """
    Augmentation class
    """

    def __init__(self, cfg, mode="train"):
        self.mode = mode

    def augment_train(self, image):
        p = 0.05
        compose_list = [
            A.CoarseDropout(max_holes=10, max_height=10, max_width=10, p=0.1),
            A.RandomBrightness(p=p, limit=(-0.4, 0.4)),
            A.RandomContrast(p=p, limit=(-0.8, 0.8)),
            A.Blur(p=p, blur_limit=5),
            A.MotionBlur(p=p, blur_limit=7),
            A.JpegCompression(
                p=p,
                quality_lower=30,
                quality_upper=100,
            ),
            A.Downscale(p=p, scale_min=0.6, scale_max=0.9),
            A.GaussNoise(p=p, var_limit=1e3),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(
                shift_limit=0.02, scale_limit=0.02, rotate_limit=5
            ),
            AddBorder(p=p),
            A.Normalize(),
            ToTensorV2(),
        ]
        comp = A.Compose(compose_list, p=1)
        return comp(image=image)["image"]

    def augment_test(self, image):
        compose_list = [A.Normalize(), ToTensorV2()]
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
