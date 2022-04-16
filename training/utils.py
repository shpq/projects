from omegaconf import OmegaConf
from pydoc import locate
from typing import Any
from PIL import Image
import numpy as np
import traceback
import importlib
import requests
import random
import os


def open_image(path, RGB=True):
    """
    Open images in RGB format
    """
    image = Image.open(path)
    if not RGB:
        return image
    rgbimg = Image.new("RGB", image.size)
    rgbimg.paste(image)
    return rgbimg


def url2image(url):
    image_bytearr = requests.get(url, stream=True).raw
    image = Image.open(image_bytearr)
    return image


def load_train_module(project: str, framework: str) -> Any:
    """
    Load train module from project folder
    """
    return load_obj(f"proj.{framework}.{project}.code_src.train.train")


def get_error_message(error) -> str:
    """
    Returns error message for error with simple structure
    """
    message = str(error.__class__.__name__)
    traceback_message = traceback.format_exc()
    if error.args:
        message += f": {error.args[0]}"
    message += f"\n{traceback_message}"
    return message


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    Copied from https://github.com/quantumblacklabs/kedro
    """

    # for python built-in types
    if not "." in obj_path:
        return locate(obj_path)

    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = (
        obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    )
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)


def seed_everything(framework, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if framework == "torch":
        import torch
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms = True
    elif framework == "tensorflow":
        import tensorflow as tf
        tf.random.set_seed(seed)


def load_model_dict(model_weights):
    output_dict = {}
    sep = "/" if "/" in model_weights else "\\"
    output_path_list = model_weights.split(sep)[:-2]
    model_cfg = output_path_list + [".hydra", "config.yaml"]
    model_cfg = "/".join(model_cfg)
    cfg = OmegaConf.load(model_cfg)
    model_module = output_path_list + ["code_src", "load_model"]
    model_module = ".".join(model_module)
    model = importlib.import_module(model_module)
    if model_weights.endswith(".h5"):
        model = model.get_model(cfg)
        model.load_weights(model_weights)
    else:
        import torch
        model = model.Model(cfg)
        model.load_state_dict(torch.load(model_weights))
    output_dict["cfg"] = cfg
    output_dict["model"] = model
    transform_module = output_path_list + ["code_src", "augmentation"]
    transform_module = ".".join(transform_module)
    transform = importlib.import_module(transform_module)
    transform = transform.Augmentation(cfg, mode="test")
    output_dict["transform"] = transform
    return output_dict
