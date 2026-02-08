"""
Configuration settings for the Track & Trajectory system.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# Base paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")


@dataclass
class PathConfig:
    """File and directory paths."""
    # Input
    videos_dir: str = os.path.join(DATA_DIR, "videos")
    default_video: str = os.path.join(DATA_DIR, "videos", "demo.mov")
    
    # Cache
    frames_dir: str = os.path.join(CACHE_DIR, "frames")
    output_dir: str = os.path.join(CACHE_DIR, "output")
    exports_dir: str = os.path.join(CACHE_DIR, "exports")
    
    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for path in [self.videos_dir, self.frames_dir, self.output_dir, self.exports_dir]:
            os.makedirs(path, exist_ok=True)


@dataclass
class ModelConfig:
    """SAM2 model configuration."""
    model_id: str = "facebook/sam2.1-hiera-tiny"
    device: str = "auto"  # "auto", "cuda", or "cpu"
    
    def get_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class HomographyConfig:
    """Homography and BEV settings."""
    resolution: int = 50  # pixels per meter
    default_bev_width: float = 10.0  # meters
    default_bev_height: float = 8.0  # meters
    default_rect_width: float = 0.5  # meters
    default_rect_height: float = 0.5  # meters


@dataclass
class VideoConfig:
    """Video processing settings."""
    default_fps: int = 30
    default_frame_stride: int = 1
    trail_length: int = 50
    accel_threshold: float = 1.0  # m/s^2


@dataclass
class UIConfig:
    """Gradio UI settings."""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False


@dataclass
class VisualizationConfig:
    """Visualization colors and styles."""
    colors_rgb: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ])
    
    @property
    def colors_bgr(self) -> List[Tuple[int, int, int]]:
        return [(c[2], c[1], c[0]) for c in self.colors_rgb]
    
    @property
    def colors_float(self) -> List[Tuple[float, float, float, float]]:
        return [(c[0]/255, c[1]/255, c[2]/255, 0.5) for c in self.colors_rgb]


@dataclass
class Config:
    """Main configuration container."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def __post_init__(self):
        self.paths.ensure_dirs()


# Default configuration instance
config = Config()
