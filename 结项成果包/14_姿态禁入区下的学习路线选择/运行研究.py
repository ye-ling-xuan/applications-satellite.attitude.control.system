"""Separate study entry point; no imports from the frozen actuator study."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent/'代码'))
from route_study.cli import main

if __name__ == '__main__':
    main()
