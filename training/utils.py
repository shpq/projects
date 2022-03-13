from typing import Any
from PIL import Image
import importlib


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


def load_train_module(project: str, framework: str) -> Any:
    """
    Load train module from project folder
    """
    return load_obj(f"proj.{framework}.{project}.code_src.train.train")

def get_error_message(error) -> str:
    """

    """
    message = str(error.__class__.__name__)
    if e.args:
        message += f": {error.args[0]}"
    return message

def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    Copied from https://github.com/quantumblacklabs/kedro
    """
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
