"""
Tracking tab handlers — SAM2 object selection, tracking, video export (Tab 3).
"""

import os
import logging

import cv2
import gradio as gr
from PIL import Image

from ..core.tracker import TrackedObject
from ..utils.drawing import overlay_mask
from . import helpers

logger = logging.getLogger(__name__)


class TrackingHandlers:
    """Handlers for the Tracking tab."""

    def __init__(self, app):
        self.app = app

    # ── Point Selection ─────────────────────────────────────────────────

    def on_track_click(self, image, point_mode, username, evt: gr.SelectData):
        """Handle click to add a tracking point (positive or negative)."""
        s = self.app._get_session(username)
        if not s or not s.video_ready:
            return image, "Process video first", ""

        x, y = evt.index
        positive = (point_mode == "Add Point")

        if s.current_obj_id not in s.objects:
            s.objects[s.current_obj_id] = TrackedObject(s.current_obj_id)
        s.objects[s.current_obj_id].add_point(x, y, positive=positive)

        mode_str = "+" if positive else "-"
        return (
            helpers.get_annotated_frame(s),
            f"{mode_str} ({x}, {y}) → Obj {s.current_obj_id}",
            helpers.get_points_summary(s),
        )

    def on_new_object(self, username):
        """Start tracking a new object."""
        s = self.app._get_session(username)
        if not s or not s.video_ready:
            return None, "Process video first", ""
        s.current_obj_id = max(s.objects.keys(), default=0) + 1
        s.objects[s.current_obj_id] = TrackedObject(s.current_obj_id)
        return (helpers.get_annotated_frame(s),
                f"New Object {s.current_obj_id}",
                helpers.get_points_summary(s))

    def on_undo_point(self, username):
        """Undo the last tracking point."""
        s = self.app._get_session(username)
        if not s or not s.video_ready:
            return None, "Process video first", ""
        if s.current_obj_id in s.objects:
            s.objects[s.current_obj_id].remove_last()
            if s.objects[s.current_obj_id].is_empty:
                del s.objects[s.current_obj_id]
        return (helpers.get_annotated_frame(s), "Undone",
                helpers.get_points_summary(s))

    def on_clear_points(self, username):
        """Clear all tracking points for every object."""
        s = self.app._get_session(username)
        if s:
            s.objects = {}
            s.current_obj_id = 1
        if not s or not s.video_ready:
            return helpers.create_placeholder_image(), "Cleared", ""
        return helpers.get_annotated_frame(s), "Cleared", ""

    # ── Run SAM2 Tracking ───────────────────────────────────────────────

    def on_run_tracking(self, frame_stride, prompt_frame, username,
                        progress=gr.Progress()):
        """Run SAM2 tracking and save overlay frames."""
        s = self.app._get_session(username)
        if not s or not s.video_ready or s.tracker is None:
            return None, "Process video first"

        if not s.objects or all(o.is_empty for o in s.objects.values()):
            return None, "No points selected"

        s.prompt_frame_idx = int(prompt_frame)
        total_frames = s.tracker.num_frames

        def prog_cb(p, msg):
            if "Frame" in msg:
                progress(p, desc=f"Tracking: {msg}")
            else:
                progress(p, desc=msg)

        trajectories = s.tracker.track(
            s.objects, s.frames_dir, prog_cb,
            keep_masks=True,
            prompt_frame_idx=s.prompt_frame_idx,
        )

        # Save overlay frames
        sample_frames = []
        saved = 0
        colors_float = [
            (c[0] / 255, c[1] / 255, c[2] / 255) for c in helpers.COLORS_RGB
        ]
        frames_to_save = [
            i for i in range(0, total_frames, frame_stride)
            if i in s.tracker.video_segments
        ]
        n_save = len(frames_to_save)

        for i, frame_idx in enumerate(frames_to_save):
            if i % 20 == 0:
                progress(0.9 + 0.1 * i / max(1, n_save),
                         desc=f"Saving: {i}/{n_save}")

            frame = helpers.load_frame(s, frame_idx)
            for obj_id, mask in s.tracker.video_segments[frame_idx].items():
                color = colors_float[(obj_id - 1) % len(colors_float)]
                frame = overlay_mask(frame, mask, color)

            out_path = os.path.join(s.output_dir,
                                    f"frame_{frame_idx:05d}.png")
            Image.fromarray(frame).save(out_path)
            saved += 1
            if len(sample_frames) < 8:
                sample_frames.append(frame)

        # Free mask memory after saving overlay frames
        s.tracker.clear_segments()

        progress(1.0, desc="Done")
        traj_info = " | ".join(
            [f"Obj{oid}:{len(t)}pts" for oid, t in trajectories.items()]
        )
        return sample_frames, f"{saved} frames saved. {traj_info}"

    # ── Export Tracked Video ────────────────────────────────────────────

    def on_generate_video(self, fps, username, progress=gr.Progress()):
        """Generate MP4 from tracked overlay frames."""
        s = self.app._get_session(username)
        if not s:
            return None, "Not logged in"

        frames = sorted(
            f for f in os.listdir(s.output_dir) if f.endswith('.png')
        )
        if not frames:
            return None, "No frames. Run tracking first."

        first = cv2.imread(os.path.join(s.output_dir, frames[0]))
        if first is None:
            return None, f"Cannot read: {frames[0]}"

        h, w = first.shape[:2]
        output_path = os.path.join(s.exports_dir, "tracked_video.mp4")
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
        )

        n = len(frames)
        written = 0
        for i, f in enumerate(frames):
            img = cv2.imread(os.path.join(s.output_dir, f))
            if img is not None:
                out.write(img)
                written += 1
            if i % 30 == 0:
                progress(i / n, desc=f"Writing: {i}/{n}")

        out.release()
        progress(1.0, desc="Done")
        return output_path, f"{written} frames @ {fps}fps"
