"""Configurable action-convention adapter.

Measured dataset convention (see docs/data_audit.md):
  dims 0-5 : incremental pose command on a quantized grid, |v| <= 0.9375
  dim 6    : gripper in {0, 1}, 1 = OPEN
Official LIBERO / robosuite convention differs:
  dim 6    : gripper in {-1, +1}, +1 = OPEN

This thin layer converts between "dataset space" and whatever space the
official eval harness expects, so the convention can be flipped in one place
once the interface doc arrives.
"""
import numpy as np

DATASET = "dataset"
LIBERO_RAW = "libero_raw"


def _g_d2r(g):
    return 2.0 * np.asarray(g, dtype=np.float32) - 1.0


def _g_r2d(g):
    return (np.asarray(g, dtype=np.float32) + 1.0) / 2.0


def to_space(action, space):
    """dataset space -> target space"""
    a = np.asarray(action, dtype=np.float32).copy()
    if space == DATASET:
        return a
    if space == LIBERO_RAW:
        a[..., 6] = _g_d2r(a[..., 6])
        return a
    raise ValueError("unknown space: %s" % space)


def from_space(action, space):
    """target space -> dataset space"""
    a = np.asarray(action, dtype=np.float32).copy()
    if space == DATASET:
        return a
    if space == LIBERO_RAW:
        a[..., 6] = _g_r2d(a[..., 6])
        return a
    raise ValueError("unknown space: %s" % space)
