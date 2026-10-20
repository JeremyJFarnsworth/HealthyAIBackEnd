"""Ensures ``src/`` is importable as the ``healthy`` package during tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
