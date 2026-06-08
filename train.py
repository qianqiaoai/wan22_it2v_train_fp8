import argparse
import os
import sys

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
    parser.add_argument("--swanlab-logdir", type=str, default="")
    parser.add_argument("--swanlab-experiment-name", type=str, default="")
    parser.add_argument("--disable-swanlab", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.logdir = args.logdir or getattr(config, "logdir", "logs/rectified_flow_finetune")
    config.swanlab_logdir = args.swanlab_logdir or getattr(config, "swanlab_logdir", "")
    config.swanlab_experiment_name = (
        args.swanlab_experiment_name
        or getattr(config, "swanlab_experiment_name", "")
        or os.path.basename(config.logdir.rstrip("/"))
    )
    config.disable_swanlab = (
        args.disable_swanlab
        or args.disable_wandb
        or bool(getattr(config, "disable_swanlab", False))
        or bool(getattr(config, "disable_wandb", False))
    )
    config.config_name = os.path.basename(args.config_path).rsplit(".", 1)[0]

    trainer = RectifiedFlowTrainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
