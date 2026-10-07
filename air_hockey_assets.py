"""Compatibility import; canonical geometry lives in envs._air_hockey.assets."""
import sys
from envs._air_hockey import assets as _implementation
sys.modules[__name__] = _implementation
