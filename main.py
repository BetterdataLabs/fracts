import argparse
import logging
import os
import random
import warnings
from typing import Any, Dict

import numpy as np
import torch
import yaml

from fracts import FracTS, TSData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action="store_true", default=False)
    parser.add_argument("--generate", "-g", action="store_true", default=False)
    parser.add_argument("--output-dir", "-o", type=str, default="./output")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="./data",
        help="The timeseries dataset directory, where each file is a timeseries and "
        "file names are IDs.",
    )
    parser.add_argument(
        "--static-data-path",
        "-s",
        type=str,
        default=None,
        help="The static data path, if existing.",
    )
    parser.add_argument(
        "--static-id",
        "-i",
        type=str,
        default=None,
        help="The static ID column name in static data.",
    )
    parser.add_argument(
        "--stratify",
        "-f",
        type=str,
        default=None,
        help="The stratify column name in static data when split data.",
    )
    parser.add_argument("--config", "-c", type=str, default="./config/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=None,
        help="The number of samples to generate (same as training set if not provided).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "-p",
        type=str,
        default=None,
        help="The checkpoint directory to load the model from.",
    )
    return parser.parse_args()


def edit_dict(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict):
            base[k] = edit_dict(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args.output_dir = os.path.join(*os.path.split(args.output_dir))
    args.data_dir = os.path.join(*os.path.split(args.data_dir))
    args.static_data_path = (
        os.path.join(*os.path.split(args.static_data_path))
        if args.static_data_path is not None
        else None
    )
    args.config = os.path.join(*os.path.split(args.config))

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if "__base__" in config:
        with open(config["__base__"], "r") as f:
            base = yaml.safe_load(f)
        config = edit_dict(base, config)

    os.makedirs(args.output_dir, exist_ok=True)
    return config


def train(args: argparse.Namespace) -> int:
    config = prepare(args)
    data = TSData(args.data_dir, args.static_data_path, args.static_id)
    os.makedirs(args.output_dir, exist_ok=True)
    train_data, test_data = data.split(
        config.get("test_size", 0.2),
        args.stratify,
        os.path.join(args.output_dir, "data"),
    )
    torch.save(train_data, os.path.join(args.output_dir, "train_data.pkl"))
    torch.save(test_data, os.path.join(args.output_dir, "test_data.pkl"))
    model = FracTS(args.output_dir, **config.get("general", {}))
    model.train(train_data, args.checkpoint_dir, **config.get("train", {}))
    return len(train_data)


def generate(args: argparse.Namespace) -> TSData:
    config = prepare(args)
    model = FracTS(args.output_dir, **config.get("general", {}))
    out = model.sample(
        args.n_samples,
        os.path.join(args.output_dir, "generated"),
        static_file=args.static_data_path,
        static_id=args.static_id,
        **config.get("generate", {})
    )
    torch.save(out, os.path.join(args.output_dir, "generated.pkl"))
    return out


def main():
    warnings.filterwarnings("ignore")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logging.basicConfig(format="%(name)s [%(levelname)s][%(asctime)s]: %(message)s")
    args = parse_args()
    if args.train:
        size = train(args)
        if args.n_samples is None:
            args.n_samples = size
    if args.generate:
        if args.n_samples is None:
            args.n_samples = len(
                torch.load(os.path.join(args.output_dir, "train_data.pkl"))
            )
        gen = generate(args)


if __name__ == "__main__":
    main()
