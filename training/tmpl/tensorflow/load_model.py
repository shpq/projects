import tensorflow as tf
from utils import load_obj


def get_model(cfg):
    """
    Create tf.keras model with functional API to convert further into tflite
    """
    image_size = cfg.project.training.size
    model_cfg = cfg.model.tensorflow
    model = load_obj(model_cfg.class_name)(
        **model_cfg.params, input_shape=(*image_size, 3)
    )
    x = model.output
    x = tf.keras.layers.Dense(
        cfg.project.dataset.num_classes, activation="softmax"
    )(x)
    return tf.keras.Model(model.inputs, x)
