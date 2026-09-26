"""Cluster stand-in for `pynput.keyboard`: importable, never delivers keys."""
import enum


class Key(enum.Enum):
    esc = "esc"
    space = "space"
    enter = "enter"


class KeyCode:
    def __init__(self, char=None):
        self.char = char

    @classmethod
    def from_char(cls, char):
        return cls(char)


class Listener:
    """Accepts the real signature; start()/join() do nothing (no keyboard)."""

    def __init__(self, on_press=None, on_release=None, **kwargs):
        self.on_press, self.on_release = on_press, on_release

    def start(self):
        return None

    def stop(self):
        return None

    def join(self, timeout=None):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
