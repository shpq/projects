from proj.tensorflow.human_segmentation.code_src.datagenerator import (
    DataGenerator,
)
from proj.tensorflow.human_segmentation.code_src.utils import (
    segmentation2mobile,
)
from proj.tensorflow.human_segmentation.code_src.load_model import get_model
from proj.tensorflow.human_segmentation.code_src.losses import Loss
from telegram_notifier import bot
from store import Store
import tensorflow as tf
from tqdm import tqdm
import os

modes = ["train", "test"]


def train(cfg):
    loss = Loss(cfg)
    data_generators = {m: DataGenerator(cfg, m) for m in modes}
    model = get_model(cfg)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.project.training.lr)

    model.build(
        (cfg.project.training.batch_size, *cfg.project.training.size, 3)
    )

    if cfg.general.print_model:
        model.summary()

    pretrained_path = cfg.project.model.pretrained_path
    if pretrained_path is not None:
        model.load_weights(pretrained_path)

    os.makedirs(cfg.training.checkpoints_path, exist_ok=True)

    train_model(cfg, model, data_generators, optimizer, loss)


def train_model(cfg, model, data_generators, optimizer, criterion):
    @tf.function
    def train_step(images, masks, trimap, send=False):
        with tf.GradientTape() as tape:

            pred_fusion, pred_global, pred_local = model(images, training=True)
            loss_value = criterion(
                tf.cast(images, tf.float32),
                tf.cast(masks, tf.float32),
                pred_fusion,
                pred_global,
                pred_local,
                tf.cast(trimap, tf.float32),
            )
            loss_value, semantic_loss, alpha_loss, fusion_loss = loss_value
        if send:
            bot.send_images(
                [images, pred_fusion, pred_global, pred_local],
                [[-1, 1], [0, 1], [0, 1], [0, 1]],
                size=cfg.project.training.size,
                save_path=[
                    cfg.save_info.image_path,
                    f"epoch_{epoch}",
                    mode,
                    ind,
                ],
            )

        gradients = tape.gradient(loss_value, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_value, semantic_loss, alpha_loss, fusion_loss

    @tf.function
    def test_step(images, masks, trimap, send=False):
        pred_fusion, pred_global, pred_local = model(images)
        loss_value = criterion(
            tf.cast(images, tf.float32),
            tf.cast(masks, tf.float32),
            pred_fusion,
            pred_global,
            pred_local,
            tf.cast(trimap, tf.float32),
        )
        if send:
            bot.send_images(
                [images, pred_fusion, pred_global, pred_local],
                [[-1, 1], [0, 1], [0, 1], [0, 1]],
                size=cfg.project.training.size,
                save_path=[cfg.save_info.image_path, f"epoch_{epoch}", mode],
            )
        loss_value, semantic_loss, alpha_loss, fusion_loss = loss_value
        return loss_value, semantic_loss, alpha_loss, fusion_loss

    framework = cfg.project.framework
    st = Store(framework=framework)
    log_values = ["loss", "semantic_loss", "alpha_loss", "fusion_loss"]
    save_values = ["loss", "fusion_loss"]
    send_images_every = cfg.project.training.send_images_every
    for epoch in range(cfg.project.training.epochs):
        for mode in modes:
            maxlen = 200 if mode == "train" else None
            st.reset(maxlen=maxlen)
            st.add_value(mode), st.add_value(epoch)
            step = train_step if mode == "train" else test_step
            generator = data_generators[mode]
            pbar = tqdm(enumerate(generator), total=len(generator))
            for ind, chunk in pbar:
                images, masks, trimap = chunk
                send = ind % send_images_every == 0 and ind != 0
                if send:
                    tf.config.run_functions_eagerly(True)
                loss, semantic_loss, alpha_loss, fusion_loss = step(
                    images, masks, trimap, send=send
                )
                if send:
                    tf.config.run_functions_eagerly(False)
                st.add_value(loss), st.add_value(semantic_loss)
                st.add_value(alpha_loss), st.add_value(fusion_loss)

                # set description
                pbar.set_description(st.training_description)

            st.save_global()

            if mode == "test":
                name = st.get_output_string("filename", *save_values)
                filepath = os.path.join(cfg.training.checkpoints_path, name)
                message = st.get_output_string("message", *log_values)
                segmentation2mobile(
                    model, filepath, generator, cfg, norm=[-1, 1], coreml=False
                )
                bot.send_message(message)
                bot.send_plots(st.get_global(log_values))
                model.save_weights(filepath)
