"""Execute the separate actuator study; no previous artifacts are modified."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'代码'))

if __name__ == '__main__':
    from actuator_study.cli import main
    main()
