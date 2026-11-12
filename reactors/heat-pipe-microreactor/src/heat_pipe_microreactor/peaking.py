"""Calculate pin-power peaking from a three-dimensional mesh tally.

Each mesh column is assigned to its nearest fuel pin. The result is the
maximum pin power divided by the mean power over all fuel pins.
"""

import numpy as np


def _pin_layout(cfg) -> list:
    """Return ``(x, y, ring)`` for each fuel pin, in centimeters."""
    from .geometry import generate_hex_positions_aziz
    pitch = cfg["core"]["pitch_cm"]
    return [(x, y, ring) for x, y, t, ring in generate_hex_positions_aziz(pitch)
            if t == "fuel"]


def _voxel_centers_xy(power_shape, cfg):
    """Return x/y voxel centers in centimeters for the OpenMC tally domain.

    The mesh covers ``[-2 * hex_side_cm, 2 * hex_side_cm]`` in x and y.
    """
    nx, ny = power_shape[0], power_shape[1]
    half = cfg["core"]["hex_side_cm"] * 2.0
    dx = 2.0 * half / nx
    dy = 2.0 * half / ny
    xs = -half + (np.arange(nx) + 0.5) * dx
    ys = -half + (np.arange(ny) + 0.5) * dy
    return xs, ys


def pin_power_peaking(power: np.ndarray, cfg: dict) -> dict:
    """Calculate the fuel-pin peaking factor.

    Mesh columns are assigned to the nearest fuel pin, then summed over z.
    Since the result is a maximum-to-mean ratio, it is independent of scale.

    Parameters
    ----------
    power : np.ndarray (nx, ny, nz)
        Power field, in watts per voxel or power density.
    cfg : dict
        Reactor configuration.

    Returns
    -------
    dict
        A mapping with the peaking factor, hottest pin, and mean relative
        power by ring.
    """
    px, py, rings, pin_power = _sum_pin_powers(power, cfg)
    mean_p = pin_power.mean()
    if mean_p <= 0:
        return {"pin_peaking": 1.0,
                "hottest": {"idx": 0, "x": float(px[0]), "y": float(py[0]),
                            "ring": int(rings[0]), "rel_power": 1.0},
                "ring_mean": {}}

    norm = pin_power / mean_p
    argmax = int(np.argmax(norm))
    ring_mean = {}
    for ring in sorted(set(rings.tolist())):
        ring_mean[int(ring)] = float(norm[rings == ring].mean())

    return {
        "pin_peaking": float(norm[argmax]),
        "hottest": {"idx": argmax, "x": float(px[argmax]),
                    "y": float(py[argmax]), "ring": int(rings[argmax]),
                    "rel_power": float(norm[argmax])},
        "ring_mean": ring_mean,
    }


def _sum_pin_powers(power: np.ndarray, cfg: dict):
    """Sum each mesh column into its nearest fuel pin.

    Returns
    -------
    (px, py, rings, pin_power) : arrays (n_pins,)
    """
    power = np.asarray(power, dtype=float)
    nx, ny, nz = power.shape
    xs, ys = _voxel_centers_xy(power.shape, cfg)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    power_xy = power.sum(axis=2)  # Sum power along z for each x/y column.

    pins = _pin_layout(cfg)
    if not pins:
        raise ValueError("No fuel pins were found in the reactor layout.")

    px = np.array([p[0] for p in pins])
    py = np.array([p[1] for p in pins])
    rings = np.array([p[2] for p in pins])

    d2 = ((X[..., None] - px[None, None, :]) ** 2 +
          (Y[..., None] - py[None, None, :]) ** 2)
    nearest = np.argmin(d2, axis=2)  # (nx, ny)

    pin_power = np.zeros(len(pins))
    flat_xy = power_xy.ravel()
    flat_nearest = nearest.ravel()
    for ip in range(len(pins)):
        pin_power[ip] = flat_xy[flat_nearest == ip].sum()
    return px, py, rings, pin_power


def compute_pin_powers(power: np.ndarray, cfg: dict) -> dict:
    """Return normalized pin powers and their positions for plotting."""
    px, py, rings, pin_power = _sum_pin_powers(power, cfg)
    norm = pin_power / pin_power.mean() if pin_power.mean() > 0 else pin_power
    return {"x": px, "y": py, "ring": rings, "pin_power": norm,
            "peaking": float(norm.max()) if norm.size else 1.0}
