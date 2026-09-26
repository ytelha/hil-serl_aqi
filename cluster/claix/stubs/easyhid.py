"""Cluster stand-in for `easyhid` (USB HID; SpaceMouse). See README.md."""


class HIDException(Exception):
    pass


class Enumeration:
    def __init__(self, *args, **kwargs):
        raise HIDException(
            "no USB HID devices on the cluster: this is the headless stand-in "
            "from cluster/claix/stubs (a SpaceMouse run needs a real machine)")
