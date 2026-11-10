import unittest
import numpy as np
import torch

from src.data import generate_split, split_hash
from src.metrics import binary_metrics
from src.model import SmallUNet


class PipelineTests(unittest.TestCase):
    def test_data_contract(self):
        x, y, labels, metadata = generate_split(120, 2026)
        self.assertEqual(x.shape, (120, 128, 128))
        self.assertEqual(set(np.unique(y)), {0, 1})
        self.assertTrue((x[y == 0] <= 55).all())
        self.assertTrue((x[y == 1] >= 200).all())
        self.assertEqual(np.bincount(labels).tolist(), [40, 40, 40])
        self.assertTrue((y.sum((1, 2)) > 0).all())
        self.assertTrue((y[:, [0, -1], :] == 0).all())
        self.assertTrue((y[:, :, [0, -1]] == 0).all())
        self.assertEqual(split_hash(x, y, labels), split_hash(*generate_split(120, 2026)[:3]))
        self.assertNotEqual(split_hash(x, y, labels), split_hash(*generate_split(120, 2027)[:3]))
        for r in metadata:
            if r["shape"] == "circle": self.assertTrue(10 <= r["radius"] <= 25)
            if r["shape"] == "square": self.assertTrue(12 <= r["half_side"] <= 25)
            if r["shape"] == "triangle": self.assertTrue(30 <= r["vertex_span"] <= 60)

    def test_metrics_known_counts(self):
        m = binary_metrics([1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0])
        self.assertEqual([m[k] for k in ["tp", "tn", "fp", "fn"]], [2, 2, 1, 1])
        self.assertAlmostEqual(m["accuracy"], 4/6)
        self.assertAlmostEqual(m["iou"], 2/4)

    def test_forward_and_gradient(self):
        model = SmallUNet()
        x = torch.rand(2, 1, 128, 128)
        target = (x >= 0.5).float()
        logits = model(x)
        self.assertEqual(logits.shape, target.shape)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
