import tensorflow_addons as tfa
from utils import load_obj
import tensorflow as tf


def get_model(cfg):
    """
    Create tf.keras model with functional API to convert further into tflite
    """
    image_size = cfg.project.training.size
    model_cfg = cfg.model
    src = tf.keras.layers.Input(shape=(*image_size, 3))
    model = load_obj(model_cfg.class_name)(
        **model_cfg.params, input_tensor=src
    )
    shape_to_name = {}
    for layer in model.layers[1:]:
        shape = layer.output_shape[1]
        if image_size[0] % shape == 0:
            shape_to_name[shape] = layer.name
    layer_names = list(shape_to_name.values())

    outputs = []
    for layer in model.layers:
        if layer.name in layer_names:
            outputs.append(layer.output)

    backbone_channels = cfg.project.model.backbone_channels
    hr_channels = cfg.project.model.hr_channels

    pred_global, lr8, lr2, lr4 = lr_branch(outputs, backbone_channels)

    pred_local, hr2 = hr_branch(
        src, lr2, lr4, lr8, backbone_channels, hr_channels
    )

    pred_fusion = fusion_branch(src, hr2, lr8, backbone_channels, hr_channels)
    return tf.keras.Model(model.inputs, [pred_fusion, pred_global, pred_local])


def resize(inp, scale=2, size=None):
    if size is None:
        s = int(scale * inp.shape[1])
        s = (s, s)
    else:
        s = size
    return tf.image.resize(inp, size=s)


def cbam_block(x, filters, ratio=1 / 4):
    _avg = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    _max = tf.keras.layers.GlobalMaxPooling2D(keepdims=True)(x)
    conv_1 = tf.keras.layers.Conv2D(
        int(filters * ratio),
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
        use_bias=True,
        activation=tf.nn.relu,
    )

    conv_2 = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
        use_bias=True,
    )
    conv_3 = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        strides=1,
        activation="sigmoid",
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
    )
    _avg = conv_2(conv_1(_avg))
    _max = conv_2(conv_1(_max))
    out = tf.keras.layers.Activation("sigmoid")(_avg + _max)
    x = x * out
    avg_out = tf.reduce_mean(x, axis=3)
    max_out = tf.reduce_max(x, axis=3)
    out = tf.stack([avg_out, max_out], axis=3)
    out = conv_3(out)
    x = x * out
    return x


def ConvIBNormRelu(
    x, filters, kernel_size, strides=1, with_inst=True, with_relu=True
):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same"
    )(x)
    if with_inst:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
    if with_relu:
        x = tf.keras.layers.Activation("relu")(x)
    return x


def lr_branch(outputs, backbone_channels):
    lr2, lr4, lr16 = outputs[0], outputs[1], outputs[3]

    lr16 = cbam_block(lr16, filters=lr16.shape[-1], ratio=1 / 4)
    # lr16 = psa_block(lr16, filters=lr16.shape[-1])
    lr16 = ConvIBNormRelu(lr16, backbone_channels[3], 5)
    lr8 = resize(lr16, scale=2)
    lr8 = ConvIBNormRelu(lr8, backbone_channels[2], 5)
    lr = ConvIBNormRelu(lr8, 1, 3, strides=2, with_inst=False, with_relu=False)
    pred_global = tf.keras.layers.Activation("sigmoid")(lr)
    return pred_global, lr8, lr2, lr4


def hr_branch(src, enc2, enc4, lr8, backbone_channels, hr_channels):

    img2 = resize(src, scale=1 / 2)
    img4 = resize(src, scale=1 / 4)

    enc2 = ConvIBNormRelu(enc2, hr_channels, 1)
    hr4 = ConvIBNormRelu(tf.concat([img2, enc2], -1), 2 * hr_channels, 3)

    enc4 = ConvIBNormRelu(enc4, hr_channels, 1)
    hr4 = ConvIBNormRelu(tf.concat([img4, enc4], -1), 2 * hr_channels, 3)

    lr4 = resize(lr8, scale=2)
    hr4 = ConvIBNormRelu(tf.concat([hr4, lr4, img4], -1), hr_channels, 3)

    hr2 = resize(hr4, scale=2)
    hr2 = ConvIBNormRelu(tf.concat([hr2, enc2], -1), hr_channels, 3)
    hr = resize(hr2, scale=2)
    hr = ConvIBNormRelu(tf.concat([src, hr], -1), hr_channels, 3)
    hr = psa_block(hr, filters=hr_channels)
    hr = tf.keras.layers.Conv2D(1, 1, strides=1, padding="same")(hr)
    pred_local = tf.keras.layers.Activation("sigmoid")(hr)

    return pred_local, hr2


def fusion_branch(src, hr2, lr8, backbone_channels, hr_channels):
    lr4 = resize(lr8, scale=2)
    lr4 = ConvIBNormRelu(lr4, hr_channels, 5)
    lr2 = resize(lr4, scale=2)

    f2 = ConvIBNormRelu(tf.concat([hr2, lr2], -1), hr_channels, 3)
    f = resize(f2, scale=2)
    f = ConvIBNormRelu(tf.concat([f, src], -1), int(hr_channels / 2), 3)
    f = psa_block(f, filters=int(hr_channels / 2))
    f = tf.keras.layers.Conv2D(1, 1, strides=1, padding="same")(f)
    pred_fusion = tf.keras.layers.Activation("sigmoid")(f)

    return pred_fusion



def psa_block(x, filters):

    inner_filters = filters // 2

    conv_q_left = tf.keras.layers.Conv2D(1, 1, strides=1, padding="same", use_bias=False)
    conv_v_left = tf.keras.layers.Conv2D(inner_filters, 1, strides=1, padding="same", use_bias=False)
    conv_up = tf.keras.layers.Conv2D(filters, 1, strides=1, padding="same", use_bias=False)

    conv_q_right = tf.keras.layers.Conv2D(inner_filters, 1, 1, padding="same", use_bias=False)
    conv_v_right = tf.keras.layers.Conv2D(inner_filters, 1, 1, padding="same", use_bias=False)

    # channel part
    # [B, H, W, iC]
    x_channel = conv_v_left(x)
    b_size, h_size, w_size, c_size = tf.shape(x_channel)

    # [B, H*W, iC]
    x_channel = tf.reshape(x_channel, [b_size, h_size * w_size, c_size])

    # [B, H, W, 1]
    context_mask = conv_q_left(x)

    # [B, H*W, 1]
    context_mask = tf.reshape(context_mask, [b_size, h_size * w_size, 1])

    # [B, H*W, 1]
    context_mask = tf.keras.layers.Softmax(axis=1)(context_mask)

    # [B, iC, 1]
    context_channel = tf.linalg.matmul(x_channel, context_mask, transpose_a=True)

    # [B, 1, 1, iC]
    context_channel = tf.reshape(context_channel, [b_size, 1, 1, inner_filters])

    # [B, 1, 1, C]
    context_channel = conv_up(context_channel)

    # [B, 1, 1, C]
    mask_channels = tf.keras.activations.sigmoid(context_channel)

    # spatial part

    # [B, H, W, iC]
    g_x = conv_q_right(x)
    _, h_size, w_size, _ = tf.shape(g_x)

    # [B, 1, 1, iC]
    avg_x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(g_x)

    # [B, 1, iC]
    avg_x = tf.reshape(avg_x, [b_size, 1, c_size])

    # [B, 1, iC]
    avg_x = tf.keras.layers.Softmax(axis=2)(avg_x)

    # [B, H*W, iC]
    theta_x = tf.reshape(conv_v_right(x), [b_size, h_size * w_size, inner_filters])

    # [B, 1, H*W]
    context_channel = tf.linalg.matmul(avg_x, theta_x, transpose_b=True)

    # [B, H, W, 1]
    context_channel = tf.reshape(context_channel, [b_size, h_size, w_size, 1])

    # [B, H, W, 1]
    mask_spatial = tf.keras.activations.sigmoid(context_channel)

    return x * mask_channels + x * mask_spatial
