"""Generate independent, balanced splits without external data downloads."""
import hashlib

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")
SIZE = 128


def generate_split(n, seed):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % len(SHAPES)
    rng.shuffle(labels)
    images = np.empty((n, SIZE, SIZE), dtype=np.uint8)
    masks = np.empty_like(images)
    metadata = []
    for i, label in enumerate(labels):
        canvas = Image.new("L", (SIZE, SIZE), 0)
        draw = ImageDraw.Draw(canvas)
        angle = float(rng.uniform(0, 2 * np.pi))
        record = {"index": i, "shape": SHAPES[label], "angle_rad": angle}
        if label == 0:
            radius = int(rng.integers(10, 26))
            cx, cy = rng.integers(radius + 2, SIZE - radius - 2, size=2)
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=1)
            record.update(radius=radius, center=[int(cx), int(cy)])
        else:
            if label == 1:
                half = int(rng.integers(12, 26))
                vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
                record["half_side"] = half
            else:
                angles = np.arange(3)*2*np.pi/3 + rng.uniform(-0.25, 0.25, 3)
                radii = rng.uniform(0.8, 1.2, 3)
                vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)*radii[:, None]
                span = float(rng.uniform(30, 60))
                diameter = np.linalg.norm(vertices[:, None]-vertices[None, :], axis=-1).max()
                vertices *= span / diameter
                record["vertex_span"] = span
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            vertices = vertices @ rotation.T
            offset = rng.uniform(2-vertices.min(0), SIZE-3-vertices.max(0))
            vertices = np.rint(vertices+offset).astype(int)
            draw.polygon([tuple(p) for p in vertices], fill=1)
            record["vertices"] = vertices.tolist()
        mask = np.asarray(canvas, dtype=np.uint8)
        background = rng.integers(0, 56, (SIZE, SIZE), dtype=np.uint8)
        foreground = rng.integers(200, 256, (SIZE, SIZE), dtype=np.uint8)
        images[i] = np.where(mask == 1, foreground, background)
        masks[i] = mask
        metadata.append(record)
    return images, masks, labels, metadata


def split_hash(images, masks, labels):
    digest = hashlib.sha256()
    for array in (images, masks, labels.astype(np.int64)):
        digest.update(array.tobytes())
    return digest.hexdigest()
