"""
Track & Trajectory - SAM2-based object tracking with BEV trajectory visualization.
"""

from .core import Tracker, HomographyTransform, TrajectoryProcessor
from .utils import draw_circle, draw_polygon, smooth_trajectory

__version__ = "1.0.0"
__all__ = [
    "Tracker",
    "HomographyTransform",
    "TrajectoryProcessor",
    "draw_circle",
    "draw_polygon",
    "smooth_trajectory",
]
