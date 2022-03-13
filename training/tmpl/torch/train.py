from proj.frmwrk.example_project.code_src.dataset import Dataset
from proj.frmwrk.example_project.code_src.losses import Loss
from proj.frmwrk.example_project.code_src.load_model import Model
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
    # scheduler = lr_scheduler.ReduceLROnPlateau()
    st = Store(framework=framework)
    log_values = ["loss"]
    save_values = ["loss"]
    for epoch in range(cfg.project.training.epochs):
        for mode in modes:
            maxlen = 200 if mode == "train" else None
            st.reset(maxlen=maxlen)
            st.add_value(mode), st.add_value(epoch)
            if mode == "train":
                model.train()
            elif mode == "val":
                model.eval()
            loader = dataloaders[mode]

            pbar = tqdm(enumerate(loader), total=len(loader))
            for i, (inputs, targets) in pbar:
                inputs, targets = inputs.cuda(), targets.cuda()
                with torch.set_grad_enabled(mode == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                st.add_value(loss)
                # measure accucary
                # accuracy = 0

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
                # scheduler.step(loss)
