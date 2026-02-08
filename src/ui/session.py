"""
Per-user session state.

Isolates all mutable state between concurrent Gradio users.
"""

import os
from typing import Optional

from ..core.tracker import Tracker, TrackedObject
from ..core.homography import HomographyTransform
from ..core.trajectory import TrajectoryProcessor
from ..core.lap_detector import Lap


class UserSession:
    """Per-user session state. Each concurrent user gets their own instance."""

    def __init__(self, username: str, frames_dir: str, output_dir: str,
                 exports_dir: str, model_id: str):
        self.username = username
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.exports_dir = exports_dir
        self.model_id = model_id

        self.video_path: Optional[str] = None
        self.tracker: Optional[Tracker] = None
        self.homography = HomographyTransform()
        self.traj_processor: Optional[TrajectoryProcessor] = None

        self.objects: dict[int, TrackedObject] = {}
        self.current_obj_id = 1
        self.video_ready = False
        self.video_fps = 30.0
        self.prompt_frame_idx = 0

        # Start / Finish line (two image-space points)
        self.start_finish_line: list[list[float]] = []   # [[x1,y1], [x2,y2]]
        self.detected_laps: dict[int, list[Lap]] = {}    # obj_id → laps
        self.selected_lap: int = 0                        # 0 = all laps

        for d in [frames_dir, output_dir, exports_dir]:
            os.makedirs(d, exist_ok=True)
