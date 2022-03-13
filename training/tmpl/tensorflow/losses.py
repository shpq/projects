import tensorflow as tf


class Loss:
    def __init__(self, cfg):
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    def __call__(self, target, output):
        return self.loss(target, output)
