"""
data_analysis: segmentation + morphology analysis framework (see ../README.md).

Every module in this package still uses bare, same-directory imports
internally (``from utils_segmentation import X`` rather than
``from data_analysis.utils_segmentation import X``) -- every script here
is also meant to be run directly (``cd data_analysis && python segmentation.py``),
where Python puts the script's own directory on ``sys.path`` automatically
and a bare import is the only thing that works (a relative import like
``from .utils_segmentation import X`` would fail outright when the module is
executed as ``__main__``, which is exactly what happens on direct invocation).

The one line below is what keeps those same bare imports working when this
package is imported normally instead (``pip install`` this repo, then
``from data_analysis.eval_metrics import evaluate_image`` from another
project, e.g. the manuscript repo) -- inserting this package's own
directory onto ``sys.path`` as soon as the package itself is imported means
every submodule's internal bare imports resolve exactly the same way
regardless of which of the two invocation styles brought them in.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
