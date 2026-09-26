"""Cluster stand-in for `pyrealsense2` (Intel RealSense). See ../README.md.

`import pyrealsense2 as rs` works; any attribute use (rs.pipeline(), ...)
raises, since there are no cameras on the cluster.
"""


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    raise RuntimeError(
        f"pyrealsense2.{name}: no RealSense cameras on the cluster (headless "
        "stand-in from cluster/claix/stubs)")
