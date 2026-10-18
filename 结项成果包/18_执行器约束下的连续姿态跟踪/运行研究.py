"""Stable numerical-library initialization for the tested Windows runtime.

Loading SciPy's numerical library before Torch avoids the duplicate OpenMP
initialization observed in the supplied Anaconda installation.  No runtime
collision override such as KMP_DUPLICATE_LIB_OK is used.
"""
from pathlib import Path
import sys
import numpy as np
import scipy.linalg
import scipy.integrate

# Initialize the already installed NumPy BLAS runtime before Torch is loaded.
# Import order alone is insufficient because BLAS initializes lazily.
_runtime_warmup = float(np.trace(np.eye(100) @ np.eye(100)))

sys.path.insert(0, str(Path(__file__).resolve().parent/'代码'))
from study import main

if __name__ == '__main__':
    main()
