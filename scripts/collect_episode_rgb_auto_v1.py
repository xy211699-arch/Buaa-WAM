"""Compatibility import for the final task runtime; use collect_data.sh to collect."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs._air_hockey import runtime as _implementation
if __name__ == "__main__":
    from envs._air_hockey.expert import main
    main()
else:
    sys.modules[__name__] = _implementation
