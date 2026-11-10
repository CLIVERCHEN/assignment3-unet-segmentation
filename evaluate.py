"""Evaluate the saved checkpoint without retraining or changing original results."""
import argparse
import json
from pathlib import Path

import torch

from src.data import SHAPES, generate_split, split_hash
from src.metrics import binary_metrics
from src.model import SmallUNet
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("results/model.pt"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    test = generate_split(config["n_test"], config["test_seed"])
    assert split_hash(*test[:3]) == config["test_sha256"], "Test data differs from the recorded run"
    model = SmallUNet().to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    metrics, predictions = evaluate(model, test[0], test[1], config["batch_size"], args.device)
    result = {"device": args.device, "test": metrics,
              "by_shape": {s: binary_metrics(predictions[test[2] == i], test[1][test[2] == i])
                           for i, s in enumerate(SHAPES)}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
