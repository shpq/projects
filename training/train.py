from utils import load_train_module
from telegram_notifier import bot
from omegaconf import DictConfig
from typing import Any
import warnings
import logging
import shutil
import hydra
import os


warnings.filterwarnings("ignore")

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-7s [LINE:%(lineno)-4d] \
[PID: %(process)-3d] # %(filename)s %(message)s",
    level=logging.DEBUG,
)


def save_code(project: str, framework: str) -> None:
    """
    Copy project code_src dir into current run dir
    """

    # create dir for copying code into run dir
    run_code_path = os.path.join(os.getcwd(), "code_src")
    os.makedirs(run_code_path, exist_ok=True)

    # get main code dir
    proj_path = os.path.join(hydra.utils.get_original_cwd(), "proj")
    proj_code_path = os.path.join(proj_path, framework, project, "code_src")

    # copy code into run dir
    for filename in os.listdir(proj_code_path):
        if not filename.endswith(".py"):
            print(f"code saving: skipping {filename}")
            continue

        shutil.copyfile(
            os.path.join(proj_code_path, filename),
            os.path.join(run_code_path, filename),
        )


def change_project_paths(d: Any, proj_name: str, framework: str) -> DictConfig:
    """
    Replace relative project paths with OS based
    """
    if isinstance(d, DictConfig):
        for k, v in d.items():
            if isinstance(v, str):
                if k.endswith("path"):
                    sep = "/" if "/" in v else "\\"
                    dirs = os.path.join(*v.split(sep))
                    hydra_cwd = hydra.utils.get_original_cwd()
                    # some paths can be created using terminal search
                    # so we handle this situation using simple logic
                    if not dirs.startswith("proj"):
                        d[k] = os.path.join(
                            hydra_cwd, "proj", framework, proj_name, dirs
                        )
                    else:
                        d[k] = os.path.join(hydra_cwd, dirs)
            else:
                d[k] = change_project_paths(v, proj_name, framework)
        return d
    return d


@hydra.main(config_path="conf", config_name="cfg")
def run(cfg: DictConfig) -> None:
    # save code if needed
    if cfg.general.save_code:
        save_code(cfg.project.name, cfg.project.framework)

    # change project paths to OS based
    change_project_paths(cfg.project, cfg.project.name, cfg.project.framework)

    # load train module
    train_module = load_train_module(cfg.project.name, cfg.project.framework)

    # start training
    train_module(cfg)
    # set project
    bot.set_project(cfg.project.name)


# python train.py project=test model=timm
if __name__ == "__main__":
    run()
