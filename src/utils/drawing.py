"""
Drawing utilities for visualization.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]
Point = Tuple[int, int]


def draw_circle(
    img: np.ndarray,
    center: Tuple[float, float],
    radius: int,
    color: Color,
    thickness: int = -1,
    outline: Optional[Color] = None,
    outline_thickness: int = 2
) -> None:
    """Draw circle with optional outline."""
    pt = (int(center[0]), int(center[1]))
    cv2.circle(img, pt, radius, color, thickness, cv2.LINE_AA)
    if outline:
        cv2.circle(img, pt, radius + outline_thickness, outline, outline_thickness, cv2.LINE_AA)


def draw_polygon(
    img: np.ndarray,
    points: List[Tuple[float, float]],
    color: Color,
    thickness: int = 2,
    closed: bool = True
) -> None:
    """Draw polygon from points."""
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(img, [pts], closed, color, thickness, cv2.LINE_AA)


def draw_text(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Color = (255, 255, 255),
    scale: float = 0.6,
    thickness: int = 2,
    shadow: bool = False
) -> None:
    """Draw text with optional shadow."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img, text, (position[0]+1, position[1]+1), font, scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, position, font, scale, color, thickness, cv2.LINE_AA)


def smooth_trajectory(coords: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth trajectory using moving average (vectorized).
    
    Args:
        coords: Nx2 array of (x, y) coordinates
        window: Smoothing window size
        
    Returns:
        Smoothed Nx2 array
    """
    if len(coords) < window:
        return coords
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(coords)
    for i in range(coords.shape[1]):
        smoothed[:, i] = np.convolve(coords[:, i], kernel, mode='same')
    return smoothed


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay a colored mask on frame.
    
    Args:
        frame: HxWx3 image (uint8)
        mask: HxW binary mask
        color: RGB color tuple (0-1 range)
        alpha: Blend factor
        
    Returns:
        Frame with mask overlay
    """
    result = frame.astype(np.float32) / 255.0
    mask_2d = mask.squeeze().astype(bool)
    for c in range(3):
        result[:, :, c] = np.where(
            mask_2d,
            result[:, :, c] * (1 - alpha) + color[c] * alpha,
            result[:, :, c]
        )
    return (result * 255).astype(np.uint8)
