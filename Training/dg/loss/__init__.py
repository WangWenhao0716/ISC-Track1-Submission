from __future__ import absolute_import

from .triplet import SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth
from .cosfacepairwise import CosfacePairwiseLoss
__all__ = [
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'CosfacePairwiseLoss'
]
