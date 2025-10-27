"""
Axonet: A comprehensive library for neuron morphology analysis from SWC files.

This library provides tools for:
- Loading and parsing SWC files
- Representing neuron morphology as graph structures
- Computing morphological features and measurements
- Visualizing neurons in various formats
- Analyzing and comparing neuron morphologies
"""

from .core import Neuron, SWCNode
from .io import load_swc
from .analysis import MorphologyAnalyzer
from .visualization import MeshRenderer, ANSIRenderer

__version__ = "0.1.0"
__author__ = "Axonet Team"

__all__ = [
    "Neuron",
    "SWCNode", 
    "load_swc",
    "MorphologyAnalyzer",
    "MeshRenderer",
    "ANSIRenderer"
]
