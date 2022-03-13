from collections import deque, defaultdict
from utils import get_error_message
from functools import partial
from executing import Source
from fnmatch import fnmatch
from typing import Any
import logging
import inspect


class Store:
    """
    Store and process training params value.
    """

    def __init__(self, maxlen=200, framework="torch"):
        self.framework = framework
        self.add_formats = {
            "message": {"add": "{} {:.4f}", "join": "\n \t - "},
            "filename": {"add": "{}_{:.4f}", "join": "_"},
        }
        self.global_values = defaultdict(list)
        self.reset(maxlen)

    def save_general_values(self, name, value):
        """
        Save general values (epoch num or train/test mode)
        """
        if name == "epoch":
            self.epoch = value
            return value
        if name == "mode":
            self.mode = value
            return value

        return None

    def add_value(self, value) -> Any:
        """
        Store and return training value.
        Name extraction inspired by https://github.com/gruns/icecream
        """
        if self.framework == "torch":
            # avoiding import torch.is_tensor because of possible
            # resources usage conflict between tf and torch
            try:
                value_to_add = value.item()
            except Exception as e:
                message = get_error_message(e)
                logging.info(message)
                value_to_add = value
        elif self.framework == "tensorflow":
            value_to_add = value
        callFrame = inspect.currentframe().f_back
        callNode = Source.executing(callFrame).node
        source = Source.for_frame(callFrame)
        name = source.asttokens().get_text(callNode.args[0])
        general_value = self.save_general_values(name, value)
        if general_value is not None:
            return general_value

        self.values[self.mode + "_" + name].append(value_to_add)
        return value

    def reset(self, maxlen):
        """
        Reset values to default with different maxlen
        """
        self.values = defaultdict(partial(deque, maxlen=maxlen))

    def save_global(self):
        """
        Create / update global values from training values by getting average
        """
        for k, v in self.values.items():
            self.global_values[k].append(sum(v) / len(v))

    def select_global(self, filt):
        """
        Select particular global training values
        """
        return {
            k: v
            for k, v in self.global_values.items()
            if fnmatch(self.remove_prefix(k), filt)
        }

    def get_global(self, name_filter=None):
        """
        Return mean of training values
        Example: name_filter="loss_*"
        """
        if name_filter is None:
            return self.global_values
        if isinstance(name_filter, str):
            return self.select_global(name_filter)
        elif isinstance(name_filter, list):
            return [self.select_global(v) for v in name_filter]
        else:
            raise ValueError("name_filter should be None, str or list of str")

    def get_training_description(self):
        """
        Return training description for progress bar
        """
        description = "Epoch {}, {}: ".format(self.epoch, self.mode)
        description_list = []
        for _name, _value in self.values.items():
            name = self.remove_prefix(_name)
            value = sum(_value) / len(_value)
            description_list.append("{} {:.4f}".format(name, value))

        description += ", ".join(description_list)
        return description

    @property
    def training_description(self):
        return self.get_training_description()

    def remove_prefix(self, name):
        """
        Remove prefix if exists to get training value name
        """
        for mode in ["train", "test"]:
            prefix = mode + "_"
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        return name

    def get_output_string(self, mode="filename", *args):
        """
        Return string consists of training values for different tasks
        (filename / message)
        """
        ext = ".h5" if self.framework == "tensorflow" else ".pth"
        if mode == "filename":
            string = "ep_{}_".format(self.epoch)
        elif mode == "message":
            string = "Epoch {}:\n \t - ".format(self.epoch)
        to_add = []
        add_format = self.add_formats[mode]["add"]
        join_format = self.add_formats[mode]["join"]
        for arg in args:
            for n, v in self.global_values.items():
                if fnmatch(self.remove_prefix(n), arg):
                    to_add.append(add_format.format(n, v[-1]))
        string += join_format.join(to_add)
        if mode == "filename":
            string += ext
        return string
