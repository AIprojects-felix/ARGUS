"""
ARGUS Prediction Heads.

This module contains prediction head architectures:
- MultiLabelClassificationHead: For multi-label binary classification (genes, MSI, etc.)
- OrdinalClassificationHead: For ordinal regression (TMB levels)
"""

from argus.models.heads.classification import MultiLabelClassificationHead
from argus.models.heads.ordinal import OrdinalClassificationHead

__all__ = [
    "MultiLabelClassificationHead",
    "OrdinalClassificationHead",
]
