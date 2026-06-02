import argparse
import os
import sys

import wandb
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from trainer import RectifiedFlowTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--wandb-save-dir", type=str, default="")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.logdir = args.logdir or getattr(config, "logdir", "logs/rectified_flow_finetune")
    config.wandb_save_dir = args.wandb_save_dir or getattr(config, "wandb_save_dir", "")
    config.disable_wandb = args.disable_wandb or bool(getattr(config, "disable_wandb", False))
    config.config_name = os.path.basename(args.config_path).rsplit(".", 1)[0]

    trainer = RectifiedFlowTrainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()
        wandb.finish()


if __name__ == "__main__":
    main()
