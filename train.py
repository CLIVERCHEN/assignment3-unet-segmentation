"""Train once, evaluate the final checkpoint, and save auditable evidence.

Run: python train.py --device auto
The test set is evaluated only after the fixed training schedule completes.
"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import SHAPES, generate_split, split_hash
from src.metrics import binary_metrics
from src.model import SmallUNet


def make_loader(images, masks, batch_size, shuffle=False, seed=42):
    x = torch.from_numpy(images[:, None].astype(np.float32)/255)
    y = torch.from_numpy(masks[:, None].astype(np.float32))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed),
                      num_workers=0)


@torch.inference_mode()
def evaluate(model, images, masks, batch_size, device):
    model.eval()
    predictions, losses = [], 0.0
    for x, y in make_loader(images, masks, batch_size):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses += nn.functional.binary_cross_entropy_with_logits(logits, y).item()*len(x)
        predictions.append((torch.sigmoid(logits) >= 0.5).cpu().numpy()[:, 0])
    prediction = np.concatenate(predictions)
    metrics = binary_metrics(prediction, masks)
    metrics["bce"] = losses/len(images)
    return metrics, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    train = generate_split(1000, 2026)
    test = generate_split(200, 2027)
    loader = make_loader(train[0], train[1], args.batch_size, True, args.seed)
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    config = {**vars(args), "output": str(out), "device": device,
              "train_seed": 2026, "test_seed": 2027, "selection_seed": 2028,
              "n_train": 1000, "n_test": 200, "threshold": 0.5,
              "parameters": sum(p.numel() for p in model.parameters()),
              "python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": np.__version__, "pillow": PIL.__version__,
              "platform": platform.platform(), "processor": platform.machine(),
              "train_sha256": split_hash(*train[:3]), "test_sha256": split_hash(*test[:3]),
              "train_counts": dict(zip(SHAPES, np.bincount(train[2]).tolist())),
              "test_counts": dict(zip(SHAPES, np.bincount(test[2]).tolist()))}
    (out/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    print(json.dumps(config), flush=True)
    history, epochs, update = [], [], 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            update += 1
            value = loss.item()
            loss_sum += value*len(x)
            history.append({"update": update, "epoch": epoch, "bce": value, "batch_size": len(x)})
        row = {"epoch": epoch, "update": update, "train_bce": loss_sum/1000,
               "elapsed_seconds": time.perf_counter()-start}
        epochs.append(row)
        print(json.dumps(row), flush=True)
    if device == "mps":
        torch.mps.synchronize()
    training_seconds = time.perf_counter()-start
    torch.save({"model_state": {k:v.cpu() for k,v in model.state_dict().items()},
                "config": config}, out/"model.pt")
    train_metrics, _ = evaluate(model, train[0], train[1], args.batch_size, device)
    test_metrics, predictions = evaluate(model, test[0], test[1], args.batch_size, device)
    by_shape = {shape: binary_metrics(predictions[test[2] == i], test[1][test[2] == i])
                for i, shape in enumerate(SHAPES)}
    individual = [{"index": i, "shape": SHAPES[test[2][i]],
                   **binary_metrics(predictions[i], test[1][i])} for i in range(200)]
    rng = np.random.default_rng(2028)
    selected = np.concatenate([rng.choice(np.flatnonzero(test[2] == i), 2, replace=False)
                               for i in range(3)])
    metrics = {"train": train_metrics, "test": test_metrics, "by_shape": by_shape,
               "mean_image_iou": float(np.mean([r["iou"] for r in individual])),
               "background_baseline": binary_metrics(np.zeros_like(test[1]), test[1]),
               "intensity_baseline": binary_metrics(test[0] >= 128, test[1]),
               "training_seconds": training_seconds, "updates": update,
               "selected_indices": selected.tolist()}
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    for name, rows in [("history.csv", history), ("epochs.csv", epochs), ("per_image.csv", individual)]:
        with (out/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out/"test_metadata.json").write_text(json.dumps(test[3], indent=2)+"\n")
    np.savez_compressed(out/"selected_examples.npz", images=test[0][selected],
                        masks=test[1][selected], predictions=predictions[selected],
                        labels=test[2][selected], indices=selected)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
