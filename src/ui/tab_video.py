"""
Video tab handlers — upload, process, frame extraction (Tab 1).
"""

import os
import shutil
import logging

import gradio as gr

from ..core.tracker import Tracker
from . import helpers

logger = logging.getLogger(__name__)


class VideoHandlers:
    """Handlers for the Video tab."""

    def __init__(self, app):
        self.app = app  # back-reference to AppUI for session lookup

    # ── User Video Library ──────────────────────────────────────────────

    def get_user_videos(self, username: str = None) -> list:
        """Return list of dicts describing the user's saved videos."""
        if not username:
            return []
        videos_dir = self.app._get_user_videos_dir(username)
        if not os.path.exists(videos_dir):
            return []
        videos = []
        for f in os.listdir(videos_dir):
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                path = os.path.join(videos_dir, f)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                videos.append({
                    "name": f, "path": path, "size_mb": round(size_mb, 1)
                })
        return sorted(videos,
                      key=lambda x: os.path.getmtime(x["path"]),
                      reverse=True)

    def load_user_video(self, video_name, username):
        """Load a previously saved user video."""
        s = self.app._get_session(username)
        if not s or not video_name:
            return None, "No video selected", gr.update(interactive=False)

        video_path = os.path.join(
            self.app._get_user_videos_dir(username), video_name
        )
        if not os.path.exists(video_path):
            return None, f"Video not found: {video_name}", \
                gr.update(interactive=False)

        s.video_path = video_path
        helpers.clear_frames(s)
        s.video_ready = False
        s.tracker = None
        s.objects = {}
        s.current_obj_id = 1

        return video_path, f"Loaded: {video_name}", \
            gr.update(interactive=True)

    # ── Upload ──────────────────────────────────────────────────────────

    def on_video_upload(self, video_path_or_file, username):
        """Handle video file upload from gr.Video component."""
        s = self.app._get_session(username)
        if video_path_or_file is None:
            return ("Upload a video to start",
                    gr.update(interactive=False), gr.update())

        src_path = (video_path_or_file
                    if isinstance(video_path_or_file, str)
                    else str(video_path_or_file))
        if not os.path.exists(src_path):
            return (f"File not found: {src_path}",
                    gr.update(interactive=False), gr.update())

        file_size_mb = os.path.getsize(src_path) / (1024 * 1024)
        if file_size_mb > helpers.MAX_VIDEO_SIZE_MB:
            return (
                f"File too large: {file_size_mb:.1f}MB. "
                f"Maximum: {helpers.MAX_VIDEO_SIZE_MB}MB",
                gr.update(interactive=False), gr.update()
            )

        video_name = os.path.basename(src_path)
        if username:
            videos_dir = self.app._get_user_videos_dir(username)
        else:
            videos_dir = os.path.join(self.app.root_dir, "data", "videos")
        os.makedirs(videos_dir, exist_ok=True)

        dest_path = os.path.join(videos_dir, video_name)
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            shutil.copy2(src_path, dest_path)

        if s:
            s.video_path = dest_path
            helpers.clear_frames(s)
            s.video_ready = False
            s.tracker = None
            s.objects = {}
            s.current_obj_id = 1

        saved_videos = self.get_user_videos(username)
        video_choices = [v["name"] for v in saved_videos]

        return (
            f"Ready: {video_name} ({file_size_mb:.1f}MB)",
            gr.update(interactive=True),
            gr.update(
                choices=video_choices,
                value=video_name if video_name in video_choices else None
            ),
        )

    # ── Process (extract frames + load SAM2) ────────────────────────────

    def on_process_video(self, username, progress=gr.Progress()):
        """Extract frames from the uploaded video and initialise tracker."""
        s = self.app._get_session(username)
        if not s or not s.video_path or not os.path.exists(s.video_path):
            ph = helpers.create_placeholder_image("Upload a video first")
            return (ph, "No video loaded", gr.update(interactive=False),
                    ph, gr.update(interactive=False), gr.update())

        progress(0, desc="Initializing...")
        s.tracker = Tracker(model_id=s.model_id)

        progress(0.02, desc="Clearing cache...")
        helpers.clear_frames(s)
        helpers.clear_output(s)

        progress(0.05, desc="Extracting frames...")

        def extract_progress(frac, msg):
            progress(0.05 + 0.85 * frac, desc=msg)

        try:
            s.tracker.extract_frames(
                s.video_path, s.frames_dir,
                force=True, progress_callback=extract_progress,
            )
        except ValueError as e:
            ph = helpers.create_placeholder_image("Cannot open video")
            return (ph, str(e), gr.update(interactive=False),
                    ph, gr.update(interactive=False), gr.update())

        s.video_fps = s.tracker.video_fps
        s.video_ready = True
        s.prompt_frame_idx = 0

        progress(0.92, desc="Loading SAM2...")
        _ = s.tracker.predictor          # force model load
        progress(1.0, desc="Done")

        frame = helpers.get_annotated_frame(s)
        nf = s.tracker.num_frames

        return (
            frame,
            f"{nf} frames ready ({s.video_fps:.1f} fps). "
            "Proceed to Calibration tab.",
            gr.update(interactive=True),                                     # calc_btn
            frame,                                                           # homo_img
            gr.update(interactive=True),                                     # new_btn
            gr.update(maximum=max(0, nf - 1), value=0, interactive=True),   # slider
        )

    # ── Frame Scrubber ──────────────────────────────────────────────────

    def on_frame_change(self, frame_idx, username):
        """Handle prompt-frame slider change."""
        s = self.app._get_session(username)
        if not s or not s.video_ready:
            return helpers.create_placeholder_image()
        s.prompt_frame_idx = int(frame_idx)
        return helpers.draw_frame_annotations(
            s, helpers.load_frame(s, int(frame_idx))
        )
