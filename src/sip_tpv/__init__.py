"""Utilities for converting between SIP and TPV FITS distortion conventions."""

from importlib.metadata import version

from .sip_to_pv import sip_to_pv
from .pv_to_sip import pv_to_sip

__all__ = ["sip_to_pv", "pv_to_sip", "__version__"]

__version__ = version("sip_tpv")
