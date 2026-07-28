import argparse
import logging
import os
import random
import time
import warnings
from typing import Any, Dict, Optional

# FracTS is PyTorch-only; keep Transformers from importing TensorFlow/Keras.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
import yaml

from fracts import FracTS, TSData


def _reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def _dir_size_mb(path: str) -> float:
    total = 0
    if not os.path.exists(path):
        return 0.0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total / (1024**2)


def _model_size_stats(output_dir: str, model: Optional[FracTS] = None) -> Dict[str, Any]:
    seq_dir = os.path.join(output_dir, "seq")
    best_pt = os.path.join(seq_dir, "best_model.pt")
    model_pt = os.path.join(seq_dir, "model.pt")
    ckpt = best_pt if os.path.exists(best_pt) else model_pt

    n_params = None
    if model is not None and getattr(model, "sequential_generator", None) is not None:
        n_params = sum(
            p.numel() for p in model.sequential_generator.parameters() if p.requires_grad
        )
    elif os.path.exists(ckpt):
        obj = torch.load(ckpt, map_location="cpu")
        if isinstance(obj, dict) and "model" in obj:
            n_params = sum(p.numel() for p in obj["model"].parameters())
        elif isinstance(obj, dict) and "model_state" in obj:
            n_params = sum(v.numel() for v in obj["model_state"].values())

    ckpt_mb = os.path.getsize(ckpt) / (1024**2) if os.path.exists(ckpt) else 0.0
    return {
        "n_params": n_params,
        "ckpt_path": ckpt if os.path.exists(ckpt) else None,
        "ckpt_mb": ckpt_mb,
        "seq_dir_mb": _dir_size_mb(seq_dir),
    }


def _print_stats(title: str, stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 60 + "\n")


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
    parser.add_argument("--seed", "-sd", type=int, default=42)
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=None,
        help="The number of samples to generate (same as training set if not provided).",
    )
    parser.add_argument(
        "--checkpoint_path",
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
        random_state=args.seed,
    )
    torch.save(train_data, os.path.join(args.output_dir, "train_data.pkl"))
    torch.save(test_data, os.path.join(args.output_dir, "test_data.pkl"))
    model = FracTS(args.output_dir, **config.get("general", {}))

    _reset_peak_gpu_memory()
    start = time.perf_counter()
    model.train(train_data, args.checkpoint_path, **config.get("train", {}))
    train_time = time.perf_counter() - start
    train_mem = _peak_gpu_memory_mb()
    size = _model_size_stats(args.output_dir, model)

    _print_stats(
        "FracTS TRAINING STATS",
        {
            "training time": f"{train_time:.1f} s ({train_time / 60:.2f} min)",
            "memory usage training": f"{train_mem:.1f} MiB",
            "model size (params)": (
                f"{size['n_params']:,} ({size['n_params'] / 1e6:.2f}M)"
                if size["n_params"] is not None
                else "n/a"
            ),
            "model size (checkpoint)": f"{size['ckpt_mb']:.2f} MB",
            "model size (seq dir)": f"{size['seq_dir_mb']:.2f} MB",
            "checkpoint": size["ckpt_path"],
        },
    )
    return len(train_data)


def generate(args: argparse.Namespace) -> TSData:
    config = prepare(args)
    model = FracTS(args.output_dir, **config.get("general", {}))

    _reset_peak_gpu_memory()
    start = time.perf_counter()
    out = model.sample(
        args.n_samples,
        os.path.join(args.output_dir, "generated"),
        static_file=args.static_data_path,
        static_id=args.static_id,
        checkpoint_path=args.checkpoint_path,
        **config.get("generate", {}),
    )
    gen_time = time.perf_counter() - start
    gen_mem = _peak_gpu_memory_mb()
    torch.save(out, os.path.join(args.output_dir, "generated.pkl"))
    size = _model_size_stats(args.output_dir, model)

    _print_stats(
        "FracTS GENERATION STATS",
        {
            "inference time": f"{gen_time:.1f} s ({gen_time / 60:.2f} min)",
            "memory usage generation": f"{gen_mem:.1f} MiB",
            "model size (params)": (
                f"{size['n_params']:,} ({size['n_params'] / 1e6:.2f}M)"
                if size["n_params"] is not None
                else "n/a"
            ),
            "model size (checkpoint)": f"{size['ckpt_mb']:.2f} MB",
            "n_samples": args.n_samples,
        },
    )
    return out


def main():
    warnings.filterwarnings("ignore")
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
