"""Shared operator-view compositing for the live OpenCV windows used by
`examples/record_demos.py` and `examples/train_aqi_rlpd.py`.

Layout produced by :func:`compose_obs_view`::

    [ front scene | wrist_1 ]
    [             |  (gap)  ]
    [             | wrist_2 ]
    [ tcp_pose : [...]  tcp_vel : [...]  ...   ]   <- per-key 'state' readout

It is display-only: it reads the observation and returns a new image, and never
mutates the observation that is recorded/trained on.

`cv2` is passed in by the caller so this module keeps no hard OpenCV dependency
(the live view is optional and OpenCV is imported lazily by the scripts).
"""

import numpy as np


def resolve_state_slices(env, config=None):
    """Map slices of the flattened ``state`` vector back to the proprio keys it
    was built from.

    Returns a list of ``(key, start, stop)`` tuples in flatten order, or ``[]``
    if the layout cannot be resolved (in which case the readout is skipped).
    Robust to the gymnasium wrapper stack: it walks the ``.env`` chain looking
    for the ``SERLObsWrapper``'s ``proprio_keys`` / ``proprio_space``.
    """
    def _attr(e, name):
        while e is not None:
            if name in getattr(e, "__dict__", {}):
                return e.__dict__[name]
            e = getattr(e, "env", None)
        return None

    keys = _attr(env, "proprio_keys")
    if keys is None and config is not None:
        keys = getattr(config, "proprio_keys", None)
    space = _attr(env, "proprio_space")

    slices, off = [], 0
    if keys and space is not None:
        for k in keys:
            d = int(np.prod(space[k].shape))
            slices.append((k, off, off + d))
            off += d
    return slices


def _pad_to_height(cv2, img, height):
    """Bottom-pad an image with black so it reaches ``height`` (for hconcat)."""
    if img.shape[0] >= height:
        return img
    pad = np.zeros((height - img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return cv2.vconcat([img, pad])


def _state_readout(cv2, state, state_slices, width):
    """Compact black strip printing the flat ``state`` vector grouped by key."""
    line_h = 22
    canvas = np.zeros((16 + line_h * max(1, len(state_slices)), width, 3),
                      dtype=np.uint8)
    y = 22
    for key, a, b in state_slices:
        vals = ", ".join(f"{v:+.2f}" for v in np.asarray(state[a:b]).ravel())
        cv2.putText(canvas, f"{key:<12}: [{vals}]", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y += line_h
    return canvas


def compose_obs_view(cv2, front, obs, image_keys, state_slices):
    """Compose the operator view.

    ``front`` is an already-annotated BGR scene image (the caller may have drawn
    a status banner on it). The two wrist camera inputs in ``obs`` are upscaled
    and stacked with a physical gap between them down the right-hand side, and a
    per-key readout of the numeric ``state`` is appended below. Returns the
    composited BGR image; ``obs`` is left untouched.
    """
    gap_px = 14
    tile = max(1, front.shape[0] // 2 - gap_px // 2)   # 2 tiles + gap == front h
    tiles = []
    for cam in image_keys:
        if cam not in obs:
            continue
        wi = np.asarray(obs[cam])
        if wi.ndim == 4:                    # (obs_horizon, H, W, C) -> newest
            wi = wi[-1]
        wi = cv2.cvtColor(wi, cv2.COLOR_RGB2BGR)
        wi = cv2.resize(wi, (tile, tile), interpolation=cv2.INTER_NEAREST)
        cv2.putText(wi, cam, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(wi)

    combined = front
    if tiles:
        gap = np.zeros((gap_px, tile, 3), dtype=np.uint8)   # gap between streams
        column = tiles[0]
        for t in tiles[1:]:
            column = cv2.vconcat([column, gap, t])
        h = max(front.shape[0], column.shape[0])
        combined = cv2.hconcat([_pad_to_height(cv2, front, h),
                                _pad_to_height(cv2, column, h)])

    if state_slices and "state" in obs:
        strip = _state_readout(cv2, np.asarray(obs["state"]).ravel(),
                               state_slices, combined.shape[1])
        combined = cv2.vconcat([combined, strip])
    return combined
