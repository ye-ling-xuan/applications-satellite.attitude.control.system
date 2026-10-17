"""Entry point; use --help to inspect the read-only transfer stages."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'代码'))
from transfer_study import main
if __name__=='__main__':main()
