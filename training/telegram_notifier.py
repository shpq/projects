from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt
from utils import get_error_message
from typing import Iterable
from PIL import Image
import telegram
import numpy as np
import logging
import io
import os

load_dotenv(find_dotenv())


class Telegram:
    """
    Telegram bot class. Using to send messages and photos to
    your chat bot
    """

    def __init__(self, token: str, chat_id: str):
        self.bot = telegram.Bot(token=token)
        self.chat_id = chat_id

    def send_plots(self, values):
        """
        Send plots of training values depending on epoch num
        """
        if isinstance(values, dict):
            values = [values]
        if isinstance(values, list):
            [self.send_plot(v) for v in values]

    def send_plot(self, values):
        """
        Send plot if it can be plotted
        """
        can_be_plotted = all(len(v) > 1 for _, v in values.items())
        if not can_be_plotted:
            return
        image_bytes = io.BytesIO()
        plt.figure(figsize=(7, 5))
        for k, v in values.items():
            plt.plot(v, label=k)
        plt.grid()
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(image_bytes, format="PNG")
        image_bytes.seek(0)
        self.bot.send_photo(photo=image_bytes, chat_id=self.chat_id)

    def renorm_photo(self, image, norm):
        """
        Renorm only general normalization cases (not channel-wise)
        """
        if norm is None:
            return image.astype("uint8")
        if not isinstance(norm, Iterable):
            raise ValueError("norm should be Iterable")
        if not len(norm) == 2:
            raise ValueError("norm should've 2 parameters")
        l, r = norm
        image = (image - l) * 255 / (r - l)
        return image.round().astype("uint8")

    def send_photo(self, image, norm, size=None, save_path=None):
        """
        Resize and send image with custom normalization, save in save_path
        folder with simple enumerate naming
        """
        try:
            # we cannot import torch/tf modules due to
            # memory usage conflicts
            image = image.detach().cpu()
        except Exception as e:
            message = get_error_message(e)
            logging.info(message)
        try:
            # both torch and tf have this method
            image = image.numpy()
        except Exception as e:
            message = get_error_message(e)
            logging.info(message)

        if isinstance(image, (np.ndarray, np.generic)):
            if len(image.shape) == 4:
                image = image[0]
            if image.shape[0] in (1, 3):
                image = np.moveaxis(image, 0, -1)
            image = self.renorm_photo(image, norm)
            if image.shape[-1] == 1:
                # if it's mask-like image
                image = Image.fromarray(image[..., -1], "L")
            else:
                image = Image.fromarray(image)
            if size is not None:
                image = image.resize(size)

        if not isinstance(image, Image.Image):
            err_message = "image should be tf/torch tensor, "
            err_message += "numpy array or PIL.image, "
            err_message += "but is {}".format(type(image))
            raise ValueError(err_message)

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        try:
            self.bot.send_photo(photo=image_bytes, chat_id=self.chat_id)
        except Exception as e:
            message = get_error_message(e)
            logging.debug(f"cannot send image, got exception:\n{message}")
        if save_path is not None:
            image.save(self.generate_image_path(save_path))

    def generate_image_path(self, save_path):
        if isinstance(save_path, list):
            save_path = [str(p) for p in save_path]
            save_path = os.path.join(*save_path)
        os.makedirs(save_path, exist_ok=True)
        existing_names = os.listdir(save_path)
        existing_names = [n for n in existing_names if n.endswith(".jpg")]
        if existing_names:
            existing_names = [int(n[: -len(".jpg")]) for n in existing_names]
            name = max(existing_names) + 1
        else:
            name = 0
        return os.path.join(save_path, str(name) + ".jpg")

    def send_images(self, images, norm=None, size=None, save_path=None):
        """
        Send np.array or PIL.Image images / list of images to bot
        with renormalization
        """
        if not isinstance(images, list):
            images = [images]
        if not isinstance(norm, list):
            norm = [norm] * len(images)
        if not isinstance(size, list):
            size = [size] * len(images)
        for image, n, s in zip(images, norm, size):
            self.send_photo(image, norm=n, size=s, save_path=save_path)

    def send_message(self, message=None):
        """
        Send text message to bot
        """
        logging.info(message)
        self.bot.send_message(text=message, chat_id=self.chat_id)


bot = Telegram(os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"))
