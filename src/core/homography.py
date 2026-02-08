"""
Homography transformation module for birds-eye-view conversion.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RectangleSpec:
    """Specification for a reference rectangle."""
    width: float  # meters
    height: float  # meters
    offset_x: float = 0.0  # meters from reference
    offset_y: float = 0.0  # meters from reference


class HomographyTransform:
    """
    Handles homography matrix computation and coordinate transformation
    between image space and birds-eye-view (BEV) space.
    """
    
    def __init__(self, resolution: int = 50):
        """
        Args:
            resolution: Pixels per meter in BEV space
        """
        self.resolution = resolution
        self.matrix: Optional[np.ndarray] = None
        self.inverse_matrix: Optional[np.ndarray] = None
        self.image_points: List[List[float]] = []
        self.bev_width: float = 20.0
        self.bev_height: float = 30.0
        self.offset_x: float = 0.0  # meters, positive = shift view right
        self.offset_y: float = 0.0  # meters, positive = shift view down
    
    def add_point(self, x: float, y: float) -> bool:
        """
        Add a reference point (max 8).
        
        Args:
            x, y: Image coordinates
            
        Returns:
            True if point was added
        """
        if len(self.image_points) >= 8:
            return False
        self.image_points.append([x, y])
        return True
    
    def remove_last_point(self) -> bool:
        """Remove the last added point."""
        if self.image_points:
            self.image_points.pop()
            return True
        return False
    
    def clear_points(self):
        """Clear all reference points."""
        self.image_points = []
        self.matrix = None
        self.inverse_matrix = None
    
    @property
    def point_count(self) -> int:
        return len(self.image_points)
    
    @property
    def is_ready(self) -> bool:
        """Check if enough points for homography calculation."""
        return len(self.image_points) >= 4
    
    def compute(
        self,
        rect1: RectangleSpec,
        rect2: Optional[RectangleSpec] = None,
        bev_width: float = 20.0,
        bev_height: float = 30.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Compute homography matrix from reference points.
        
        Args:
            rect1: First rectangle specification
            rect2: Optional second rectangle (requires 8 points)
            bev_width: Total BEV output width in meters
            bev_height: Total BEV output height in meters
            offset_x: Horizontal offset in meters (positive = shift view right)
            offset_y: Vertical offset in meters (positive = shift view down)
            
        Returns:
            (success, message)
        """
        if len(self.image_points) < 4:
            return False, "Need at least 4 points"
        
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        dst_w = int(bev_width * self.resolution)
        dst_h = int(bev_height * self.resolution)
        center_x, center_y = dst_w / 2, dst_h / 2
        
        # Apply offset (shift the reference point in the opposite direction)
        offset_x_px = offset_x * self.resolution
        offset_y_px = offset_y * self.resolution
        
        # Build destination points (Rect1 centered, then shifted by offset)
        r1_w_px = rect1.width * self.resolution
        r1_h_px = rect1.height * self.resolution
        r1_x = center_x - r1_w_px / 2 - offset_x_px
        r1_y = center_y - r1_h_px / 2 - offset_y_px
        
        src_pts = list(self.image_points[:4])
        dst_pts = [
            [r1_x, r1_y],
            [r1_x + r1_w_px, r1_y],
            [r1_x + r1_w_px, r1_y + r1_h_px],
            [r1_x, r1_y + r1_h_px]
        ]
        
        # Add second rectangle if available
        if rect2 and len(self.image_points) >= 8:
            src_pts.extend(self.image_points[4:8])
            r2_x = r1_x + rect2.offset_x * self.resolution
            r2_y = r1_y + rect2.offset_y * self.resolution
            r2_w_px = rect2.width * self.resolution
            r2_h_px = rect2.height * self.resolution
            dst_pts.extend([
                [r2_x, r2_y],
                [r2_x + r2_w_px, r2_y],
                [r2_x + r2_w_px, r2_y + r2_h_px],
                [r2_x, r2_y + r2_h_px]
            ])
        
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        # Calculate homography
        if len(src_pts) == 4:
            self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            method = "exact (4 points)"
        else:
            self.matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else len(src_pts)
            method = f"RANSAC ({inliers}/{len(src_pts)} inliers)"
        
        # Compute inverse for BEV to image transform
        try:
            self.inverse_matrix = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            logger.warning("Homography matrix is singular — cannot compute inverse")
            self.matrix = None
            self.inverse_matrix = None
            return False, "Homography matrix is singular. Check that points are not collinear."
        
        return True, f"Homography computed: {method}"
    
    def transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform points from image space to BEV space.
        
        Args:
            points: Nx2 array of (x, y) image coordinates
            
        Returns:
            Nx2 array of BEV coordinates, or None if no matrix
        """
        if self.matrix is None or len(points) == 0:
            return None
        
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.matrix)
        return transformed.reshape(-1, 2)
    
    def transform_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Transform a single point to BEV space."""
        result = self.transform_points(np.array([[x, y]]))
        if result is not None:
            return (float(result[0, 0]), float(result[0, 1]))
        return None
    
    def inverse_transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Transform points from BEV space back to image space."""
        if self.inverse_matrix is None or len(points) == 0:
            return None
        
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.inverse_matrix)
        return transformed.reshape(-1, 2)
    
    def to_meters(self, bev_coords: np.ndarray) -> np.ndarray:
        """Convert BEV pixel coordinates to meters."""
        return bev_coords / self.resolution
    
    def from_meters(self, meters: np.ndarray) -> np.ndarray:
        """Convert meters to BEV pixel coordinates."""
        return meters * self.resolution
    
    def warp_image(self, image: np.ndarray) -> np.ndarray:
        """
        Warp an image to birds-eye-view.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Warped BEV image
        """
        if self.matrix is None:
            raise ValueError("Homography not computed")
        
        dst_w = int(self.bev_width * self.resolution)
        dst_h = int(self.bev_height * self.resolution)
        return cv2.warpPerspective(image, self.matrix, (dst_w, dst_h))
    
    def get_bev_dimensions(self) -> Tuple[int, int]:
        """Get BEV canvas dimensions in pixels."""
        return (
            int(self.bev_width * self.resolution),
            int(self.bev_height * self.resolution)
        )
    
    def get_point_summary(self) -> str:
        """Get human-readable summary of reference points."""
        if not self.image_points:
            return "No points. Click: TL→TR→BR→BL"
        
        names = ["TL", "TR", "BR", "BL"]
        lines = ["Rect 1:"]
        for i, (x, y) in enumerate(self.image_points[:4]):
            lines.append(f"  {names[i]}: ({x:.0f}, {y:.0f})")
        
        if len(self.image_points) > 4:
            lines.append("\nRect 2:")
            for i, (x, y) in enumerate(self.image_points[4:8]):
                lines.append(f"  {names[i]}: ({x:.0f}, {y:.0f})")
        
        return "\n".join(lines)
