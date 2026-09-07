"""The land-cover CNN: a multi-label classifier over BigEarthNet v2.0.

Imports are deferred so that ``satquery`` stays importable without torch. The
classifier is an optional component of the running system, not a prerequisite
for starting it.
"""

from satquery.cnn.labels import CORINE_CLASSES, class_index, indices_to_labels
from satquery.cnn.model import build_model, inflate_stem

__all__ = [
    "CORINE_CLASSES",
    "build_model",
    "class_index",
    "indices_to_labels",
    "inflate_stem",
]
