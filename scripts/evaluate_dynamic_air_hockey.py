"""CLI wrapper; safe to use --help/check without importing a learned model."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air_hockey_eval.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
