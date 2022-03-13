from telegram_notifier import bot
import tensorflow as tf


def segmentation2tflite(model, filepath, generator, cfg):
    # import modules in func to avoid memory conflict
    from convert.mobile.tflite import convert2tflite

    filepath = filepath.replace(".h5", ".tflite")
    tflite_outputs = convert2tflite(model, filepath, generator)
    for im, out in zip(generator, tflite_outputs):
        save_path = [
            cfg.save_info.image_path,
            "tflite",
            filepath.replace(".tflite", ""),
        ]
        bot.send_images(
            [im[0][0], out], norm=[[-1, 1], [0, 1]], save_path=save_path
        )


def segmentation2coreml(model, filepath, norm):
    # import modules in func to avoid memory conflict
    from convert.mobile.coreml import convert2coreml

    filepath = filepath.replace(".h5", ".coreml")
    if norm is None:
        norm = [0, 255]
    l, r = norm
    scale = (r - l) / 255
    bias = [l] * 3

    convert2coreml(model, filepath, scale, bias)


def segmentation2mobile(
    model, filepath, generator, cfg, norm=None, tflite=True, coreml=True
):
    model2mobile = tf.keras.Model(model.inputs, model.outputs[0])

    if tflite:
        segmentation2tflite(model2mobile, filepath, generator, cfg)

    if coreml:
        segmentation2coreml(model2mobile, filepath, norm=norm)
