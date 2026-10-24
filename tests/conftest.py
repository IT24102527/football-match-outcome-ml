"""Make the repo root importable so tests can use `dashboard` and `src` (run `pytest` from the repo root)."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
