import argparse
from clipdinosam.config import load_config_with_overrides
from clipdinosam.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="key=value pairs to override config")
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.overrides)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

