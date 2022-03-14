# projects

**projects** is a light-weighted approach to do training experiments with both [Pytorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/) frameworks with power of [Hydra](https://hydra.cc/). Mostly aimed at good trade-off between code writing and configs setting.

## Using this repo you can easily:
- create Tensorflow and Pytorch projects with simple code generation approach using [create_project.py](./training/create_project.py)
- log your variables to clean your training code with [class Store](./training/store.py)
- sending training information messages / plots / images to your [telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) with [class Telegram](./training/telegram_notifier.py). Load telegram credentials via environment variables or [dotenv](https://pypi.org/project/python-dotenv/)
- generate output model filenames / training description for progress bar with built-in [class Store](./training/store.py) methods
- convert tf.keras models to coreml and tflite by [coreml](./training/convert/mobile/coreml.py) and [tflite](./training/convert/mobile/tflite.py) modules. Custom usage example in [human segmentation](./training/proj/tensorflow/human_segmentation/utils.py)
- create torch backbone models with [timm](https://github.com/rwightman/pytorch-image-models/), tensorflow by tf.keras.applications from [configs](./training/conf/model)

## Projects already done:
- [Human matting (segmentation)](./training/proj/tensorflow/human_segmentation)
- [MagFace](./training/proj/torch/magface)

## References:
- [IceCream](https://github.com/gruns/icecream) - simple variable logging way
- [Erlemar](https://github.com/Erlemar/pytorch_tempest) - good example of Hydra usage
- [kedro](https://github.com/quantumblacklabs/kedro) - importing modules from configs
