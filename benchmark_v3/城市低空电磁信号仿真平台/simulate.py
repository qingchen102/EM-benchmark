"""Source-local CLI, safe even when an old cityrf is installed elsewhere."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from cityrf.cli import main

if __name__=='__main__':
    raise SystemExit(main())
