from proj.torch.age_coral.code_src.utils import calculate_log1p
from proj.torch.age_coral.code_src.load_model import Model
from proj.torch.age_coral.code_src.dataset import Dataset
from proj.torch.age_coral.code_src.losses import Loss
from torch.utils.data import DataLoader

# from torch.optim import lr_scheduler
from telegram_notifier import bot
from store import Store
from tqdm import tqdm
import logging
import torch
import os

modes = ["train", "test"]


def train(cfg):
    datasets = {m: Dataset(cfg, m) for m in modes}
    dataloaders = {
        m: DataLoader(
            datasets[m],
            batch_size=cfg.project.training.batch_size,
            num_workers=0,
        )
        for m in modes
    }
    logging.info("loading model")
    model = Model(cfg)
    pretrained_path = cfg.project.model.pretrained_path
    framework = cfg.project.framework
    logging.info("pretrained_path")
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    optimizer = torch.optim.Adam(model.parameters(), cfg.project.training.lr)
    criterion = Loss()
    model = model.cuda()
    logging.info("making checkpoints_path")
    os.makedirs(cfg.training.checkpoints_path, exist_ok=True)
    # if scheduler_cfg != "lr":
    #     scheduler = load_obj(scheduler_cfg.class_name)(
    #             optimizer, **scheduler_cfg.params
    #         )
    # else:
    #     scheduler = None
    st = Store(framework=framework)
    log_values = ["loss", "mae_log1p"]
    save_values = ["loss", "mae_log1p"]
    for epoch in range(cfg.project.training.epochs):
        for mode in modes:
            maxlen = 200 if mode == "train" else None
            st.reset(maxlen=maxlen)
            st.add_value(mode), st.add_value(epoch)
            if mode == "train":
                model.train()
            elif mode == "test":
                model.eval()
            loader = dataloaders[mode]

            pbar = tqdm(enumerate(loader), total=len(loader))
            for i, (inputs, targets, levels) in pbar:
                inputs, targets, = inputs.cuda(), targets.cuda()
                levels = levels.cuda()
                with torch.set_grad_enabled(mode == "train"):
                    logits, probas = model(inputs)
                    loss = criterion(logits, levels)
                st.add_value(loss)
                mae_log1p = calculate_log1p(probas, targets)
                st.add_value(mae_log1p)

                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pbar.set_description(st.training_description)
            st.save_global()
            if mode == "test":
                # set name
                name = st.get_output_string("filename", *save_values)
                save_path = os.path.join(cfg.training.checkpoints_path, name)
                message = st.get_output_string("message", *log_values)
                bot.send_message(message)
                bot.send_plots(st.get_global(log_values))
                torch.save(model.state_dict(), save_path)
                # if scheduler is not None:
                #     scheduler.step(loss)
