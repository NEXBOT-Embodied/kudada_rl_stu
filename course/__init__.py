"""Repository-local Motrix sources take priority over installed sibling checkouts."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for package in ('motrix_env_core', 'motrix_env_motrixsim'):
    sys.path.insert(0, str(ROOT / 'vendor' / package / 'src'))
