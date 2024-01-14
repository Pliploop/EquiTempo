from src.training.train import *
from src.data_loading.datasets import *
from torch.utils.tensorboard import SummaryWriter
from config.full import GlobalConfig

from src.finetuning.finetuning import finetune

import argparse


test_dataset = "giantsteps"
augm = "off"
ratios = ["0.2"]

# augmentations OFF
model_name_list = [
    "pretrained_backbones/augmentations OFF/rp0.2/loss_0.0149_it_33442.pt",
]

# for eval on Hainsworth
dataset_name_list = ["hainsworth", "gtzan"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YAML config file")

    # Add an argument to accept the YAML config path
    parser.add_argument(
        "--config_path",
        required=False,
        type=str,
        help="Path to the YAML config file",
        default=None,
    )
    parser.add_argument("--dataset_name_list", nargs="+", default=None, required=False)
    parser.add_argument("--model_name", required=False, default=None)
    parser.add_argument("--override_rf", required=False, default=None)
    parser.add_argument("--resume_id", required=False, default=None)
    parser.add_argument("--resume_checkpoint", required=False, default=None)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--no-log", dest="feature", action="store_false")
    parser.add_argument("--resume_latest", default=None, required=False)

    args = parser.parse_args()

    if not args.resume_latest:
        config_path = args.config_path
        model_name = args.model_name
    else:
        config_path = os.path.join(args.resume_latest, "config.yml")
        model_name = os.path.join(args.resume_latest, "latest.pt")

    ## load config here or instanciate:
    globalconfig = GlobalConfig()  ## or from yaml if load path exists
    if args.config_path is not None:
        print(f"loading config from {config_path}")
        globalconfig.from_yaml(config_path)

    for i, model_name in enumerate(model_name_list):
        for rf in [0.0, 0.1, 0.2, 0.3, 0.4]:
            globalconfig.preprocessing_config["rf"] = rf
            globalconfig.preprocessing_config["rf_range"] = [1 - rf, 1 + rf]

            globalconfig.finetuning_config[
                "save_path"
            ] = f"checkpoints/fine_tune/var_rf/{test_dataset}/{rf}"

            print(f"finetuning from {model_name}, rf={rf}...")

            globalconfig.finetuning_config["finetuned_from"] = model_name

            finetune(
                model_name=model_name,
                dataset_name_list=dataset_name_list,
                global_config=globalconfig,
                resume_id=args.resume_id,
                resume_checkpoint=args.resume_checkpoint,
            )
