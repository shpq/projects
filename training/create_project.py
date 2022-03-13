import os
import argparse
from shutil import copytree


def create_project(name: str, framework: str, sample_dataset: bool) -> None:
    """
    Copy torch/tensorflow project classification template with a given name,
    create simple .yaml config file, add classification dataset if needed
    """
    proj_path = os.path.join("proj", framework, name)
    code_path = os.path.join(proj_path, "code_src")
    conf_path = os.path.join("conf", "project")
    os.makedirs(code_path)
    tmpl_path = os.path.join("tmpl", framework)

    print("Start copying files")
    for filename in os.listdir(tmpl_path):
        filename_path = os.path.join(tmpl_path, filename)
        with open(filename_path, "r") as f:
            code_scipt = f.read()
            code_scipt = code_scipt.replace("example_project", name)
            code_scipt = code_scipt.replace("frmwrk", framework)
        if filename.endswith(".yaml"):
            filename = name + ".yaml"
            path = conf_path
        else:
            path = code_path
        save_path = os.path.join(path, filename)
        with open(save_path, "w") as f:
            f.write(code_scipt)
    print("Done copying files")

    if sample_dataset:
        print("Start copying dataset")
        sample_dataset_path = os.path.join("tmpl", "dataset")
        sample_dataset_dest = os.path.join(proj_path, "dataset")
        copytree(sample_dataset_path, sample_dataset_dest)
        print("Done copying dataset")


# python create_project.py --framework torch --name test --sample_dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create project.")
    parser.add_argument("--framework", help="torch or tensorflow")
    parser.add_argument("--name", help="Project name")
    parser.add_argument(
        "--sample_dataset",
        action="store_true",
        help="Add sample dataset to the project?",
    )
    args = parser.parse_args()
    create_project(args.name, args.framework, args.sample_dataset)
