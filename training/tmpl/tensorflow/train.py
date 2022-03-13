from proj.frmwrk.example_project.code_src.datagenerator import DataGenerator
from proj.frmwrk.example_project.code_src.load_model import get_model
from proj.frmwrk.example_project.code_src.losses import Loss
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
    def train_step(images, target):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss_value = criterion(target, output)

        gradients = tape.gradient(loss_value, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_value

    @tf.function
    def test_step(images, target):
        output = model(images, training=True)
        loss_value = criterion(target, output)
        return loss_value

    framework = cfg.project.framework
    st = Store(framework=framework)
    log_values = ["loss"]
    save_values = ["loss"]
    for epoch in range(cfg.project.training.epochs):
        for mode in modes:
            maxlen = 200 if mode == "train" else None
            st.reset(maxlen=maxlen)
            st.add_value(mode), st.add_value(epoch)
            step = train_step if mode == "train" else test_step
            generator = data_generators[mode]
            pbar = tqdm(enumerate(generator), total=len(generator))
            for ind, chunk in pbar:
                images, target = chunk
                loss = step(images, target)
                st.add_value(loss)

                # set description
                pbar.set_description(st.training_description)

            st.save_global()

            if mode == "test":
                name = st.get_output_string("filename", *save_values)
                filepath = os.path.join(cfg.training.checkpoints_path, name)
                message = st.get_output_string("message", *log_values)
                bot.send_message(message)
                bot.send_plots(st.get_global(log_values))
                model.save_weights(filepath)
