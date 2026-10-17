"""Run independent follow-up without modifying previous research artifacts."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT/"代码"))

if __name__ == "__main__":
    from deadline_study.cli import main
    main()
