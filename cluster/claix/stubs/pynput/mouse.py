"""Cluster stand-in for `pynput.mouse` (unused by this repo)."""


class Listener:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return None

    def stop(self):
        return None

    def join(self, timeout=None):
        return None
