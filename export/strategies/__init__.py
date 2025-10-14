"""
Export Strategies - Experiment-type-specific data export implementations.

This package provides different export strategies for different experiment types:
- Stim30HzStrategy: Standard 30Hz stimulation (stimulus-aligned)
- HargreavesStrategy: Thermal sensitivity testing (event onset/withdrawal aligned)
- LickingStrategy: Licking behavior analysis (window comparison + histograms)

All strategies implement the ExportStrategy interface defined in base_strategy.py.
"""

from .base_strategy import ExportStrategy
from .stim_30hz_strategy import Stim30HzStrategy
from .hargreaves_strategy import HargreavesStrategy
from .licking_strategy import LickingStrategy

__all__ = [
    'ExportStrategy',
    'Stim30HzStrategy',
    'HargreavesStrategy',
    'LickingStrategy'
]
