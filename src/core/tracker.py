"""
SAM2-based object tracking module.

The SAM2 model is cached at the class level to avoid redundant loading
across user sessions.
"""

import os
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional, Callable, Generator
from dataclasses import dataclass, field

from sam2.build_sam import build_sam2_video_predictor_hf

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Represents an object being tracked."""
    obj_id: int
    points: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)  # 1=positive, 0=negative
    
    def add_point(self, x: float, y: float, positive: bool = True):
        self.points.append([x, y])
        self.labels.append(1 if positive else 0)
    
    def remove_last(self) -> bool:
        if self.points:
            self.points.pop()
            self.labels.pop()
            return True
        return False
    
    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0
    
    @property
    def positive_count(self) -> int:
        return sum(1 for l in self.labels if l == 1)
    
    @property
    def negative_count(self) -> int:
        return sum(1 for l in self.labels if l == 0)


class Tracker:
    """
    SAM2-based video object tracker.
    
    Handles model loading, frame extraction, and mask propagation.
    The SAM2 predictor is cached at the class level so multiple Tracker
    instances (e.g. different user sessions) share the same GPU model.
    """
    
    _model_cache: Dict[str, object] = {}  # cache_key -> predictor
    
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-tiny", device: str = "auto"):
        self.model_id = model_id
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._inference_state = None
        self.video_segments: Dict[int, Dict[int, np.ndarray]] = {}
        self.trajectories: Dict[int, List[Tuple[int, float, float]]] = {}
        self.num_frames: int = 0
        self.video_fps: float = 30.0
    
    @property
    def predictor(self):
        """Lazy-load the SAM2 predictor (shared across Tracker instances)."""
        cache_key = f"{self.model_id}:{self.device}"
        if cache_key not in Tracker._model_cache:
            logger.info("Loading SAM2 model %s on %s...", self.model_id, self.device)
            Tracker._model_cache[cache_key] = build_sam2_video_predictor_hf(
                self.model_id, device=self.device
            )
            logger.info("SAM2 model loaded successfully")
        return Tracker._model_cache[cache_key]
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        force: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> int:
        """
        Extract frames from video to JPEG files.
        
        Also reads and stores the video FPS for accurate kinematics calculations.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            force: Re-extract even if frames exist
            progress_callback: Optional callback(progress_fraction, message)
            
        Returns:
            Number of frames extracted
        """
        os.makedirs(output_dir, exist_ok=True)
        
        existing = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        if existing and not force:
            self.num_frames = len(existing)
            # Still read FPS from the video metadata
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    self.video_fps = fps
                cap.release()
            logger.info("Using %d existing frames (%.1f fps)", self.num_frames, self.video_fps)
            return self.num_frames
        
        logger.info("Extracting frames from %s...", video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Read FPS from video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.video_fps = fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(output_dir, f"{count:05d}.jpg"), frame)
            count += 1
            if progress_callback and count % 30 == 0:
                pct = count * 100 // total
                progress_callback(
                    count / max(1, total),
                    f"Extracting: {count}/{total} ({pct}%)"
                )
        cap.release()
        
        self.num_frames = count
        logger.info("Extracted %d frames (%.1f fps)", count, self.video_fps)
        return count
    
    def initialize(self, frames_dir: str):
        """Initialize tracking state for a video."""
        self._inference_state = self.predictor.init_state(video_path=frames_dir)
        self.predictor.reset_state(self._inference_state)
        self.video_segments = {}
        self.trajectories = {}
    
    def add_object(self, obj: TrackedObject, frame_idx: int = 0):
        """Add an object to track on a specific frame."""
        if obj.is_empty:
            return
        
        self.predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=frame_idx,
            obj_id=obj.obj_id,
            points=np.array(obj.points, dtype=np.float32),
            labels=np.array(obj.labels, dtype=np.int32),
        )
        self.trajectories[obj.obj_id] = []
    
    def propagate(self, keep_masks: bool = True) -> Generator[Tuple[int, Dict[int, np.ndarray]], None, None]:
        """
        Propagate tracking through video.
        
        Args:
            keep_masks: If True, store masks in video_segments for later
                       visualization. If False, only extract centroids (saves memory).
        
        Yields:
            (frame_idx, {obj_id: mask}) for each frame
        """
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(self._inference_state):
            frame_masks = {}
            
            for j, obj_id in enumerate(obj_ids):
                mask = (mask_logits[j] > 0.0).cpu().numpy()
                frame_masks[obj_id] = mask
                
                # Calculate centroid for trajectory
                centroid = self._get_mask_centroid(mask)
                if centroid and obj_id in self.trajectories:
                    self.trajectories[obj_id].append((frame_idx, centroid[0], centroid[1]))
            
            if keep_masks:
                self.video_segments[frame_idx] = frame_masks
            yield frame_idx, frame_masks
    
    def track(
        self,
        objects: Dict[int, TrackedObject],
        frames_dir: str,
        progress_callback: Optional[Callable] = None,
        keep_masks: bool = True,
        prompt_frame_idx: int = 0
    ) -> Dict[int, List[Tuple[int, float, float]]]:
        """
        Full tracking pipeline.
        
        Args:
            objects: Dictionary of TrackedObject instances
            frames_dir: Directory containing video frames
            progress_callback: Optional callback(progress, message)
            keep_masks: Whether to store masks for visualization
            prompt_frame_idx: Frame index where prompts were placed
            
        Returns:
            Trajectories dict: {obj_id: [(frame_idx, x, y), ...]}
        """
        if progress_callback:
            progress_callback(0.1, "Initializing...")
        
        self.initialize(frames_dir)
        
        if progress_callback:
            progress_callback(0.2, "Adding objects...")
        
        for obj in objects.values():
            self.add_object(obj, frame_idx=prompt_frame_idx)
        
        if progress_callback:
            progress_callback(0.3, "Tracking...")
        
        update_interval = max(1, min(5, self.num_frames // 50))
        
        for i, (frame_idx, _) in enumerate(self.propagate(keep_masks=keep_masks)):
            if progress_callback and (i % update_interval == 0 or i == self.num_frames - 1):
                progress = 0.3 + 0.6 * ((i + 1) / max(1, self.num_frames))
                progress_callback(progress, f"Frame {frame_idx + 1}/{self.num_frames}")
        
        if progress_callback:
            progress_callback(1.0, "Done")
        
        return self.trajectories
    
    def clear_segments(self):
        """Free memory by clearing stored video segments."""
        count = len(self.video_segments)
        self.video_segments.clear()
        logger.info("Cleared %d stored video segments to free memory", count)
    
    @staticmethod
    def _get_mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate centroid of binary mask."""
        mask_2d = mask.squeeze()
        if mask_2d.sum() == 0:
            return None
        y, x = np.where(mask_2d > 0)
        return (float(x.mean()), float(y.mean()))
    
    @staticmethod
    def load_frame(frames_dir: str, frame_idx: int) -> np.ndarray:
        """Load a single frame as RGB numpy array."""
        path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
        return np.array(Image.open(path))
