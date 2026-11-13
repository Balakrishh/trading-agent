"""Where upday_put_study.py saves its simulated rows (outside the repo, next to the chain data)."""
import os
from pathlib import Path

ROWS = Path(os.environ.get("UPDAY_ROWS", Path.home() / "data/research/upday_rows.pkl"))
