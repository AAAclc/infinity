"""Numpy/PIL augmentations. Deliberately excludes geometric flips
(which would break manipulation semantics)."""
import numpy as np
from PIL import Image, ImageEnhance


def aug_image(img, rng, jitter=0.2, crop_scale=0.92):
    im = Image.fromarray(img)
    if rng.random() < 0.6:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(1 - jitter, 1 + jitter))
    if rng.random() < 0.6:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(1 - jitter, 1 + jitter))
    if rng.random() < 0.4:
        im = ImageEnhance.Color(im).enhance(rng.uniform(1 - jitter, 1 + jitter))
    W, H = im.size
    if rng.random() < 0.6:
        s = rng.uniform(crop_scale, 1.0)
        nw, nh = max(int(W * s), 1), max(int(H * s), 1)
        x = rng.integers(0, W - nw + 1)
        y = rng.integers(0, H - nh + 1)
        im = im.crop((x, y, x + nw, y + nh)).resize((W, H), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def aug_state(state, rng, pos_noise=0.002, rot_noise=0.004):
    s = state.astype(np.float32).copy()
    s[:3] += rng.normal(0.0, pos_noise, 3).astype(np.float32)
    s[3:6] += rng.normal(0.0, rot_noise, 3).astype(np.float32)
    return s
