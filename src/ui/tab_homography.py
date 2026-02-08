"""
Calibration / Homography tab handlers (Tab 2).

Includes homography point selection **and** start/finish-line drawing.
"""

import os
import json
import logging

import cv2
import gradio as gr

from ..core.homography import RectangleSpec
from ..core.trajectory import TrajectoryProcessor
from . import helpers

logger = logging.getLogger(__name__)


class HomographyHandlers:
    """Handlers for the Calibration tab."""

    def __init__(self, app):
        self.app = app

    # ── Unified click dispatcher ─────────────────────────────────────────

    def on_homo_click(self, image, mode, username, evt: gr.SelectData):
        """
        Dispatch image clicks based on the current calibration mode.
        """
        if mode == "Start/Finish Line":
            return self._on_line_click(image, username, evt)
        return self._on_homo_point_click(image, username, evt)

    # ── Homography Point Selection ───────────────────────────────────────

    def _on_homo_point_click(self, image, username, evt: gr.SelectData):
        """Handle click to add a homography calibration point."""
        s = self.app._get_session(username)
        if not s or not s.video_ready:
            return image, "Process video first", ""

        x, y = evt.index
        if not s.homography.add_point(x, y):
            return (helpers.get_annotated_frame(s),
                    "Max 8 points",
                    s.homography.get_point_summary())

        n = s.homography.point_count
        status = f"Added R{1 if n <= 4 else 2}-P{((n - 1) % 4) + 1}" \
                 f" ({x:.0f}, {y:.0f})"
        if n in (4, 8):
            status += f" — {n} points ready"
        return (helpers.get_annotated_frame(s), status,
                s.homography.get_point_summary())

    def on_clear_homo(self, username):
        """Clear all homography calibration points."""
        s = self.app._get_session(username)
        if s:
            s.homography.clear_points()
        if not s or not s.video_ready:
            return helpers.create_placeholder_image(), "Cleared", ""
        return helpers.get_annotated_frame(s), "Cleared", ""

    def on_undo_homo(self, username):
        """Undo the last homography calibration point."""
        s = self.app._get_session(username)
        if s:
            s.homography.remove_last_point()
        if not s or not s.video_ready:
            return helpers.create_placeholder_image(), "Undone", ""
        return (helpers.get_annotated_frame(s), "Undone",
                s.homography.get_point_summary())

    def on_add_manual_homo(self, x, y, username):
        """Add a homography point via manual coordinate entry."""
        s = self.app._get_session(username)
        if s:
            s.homography.add_point(float(x), float(y))
        if not s or not s.video_ready:
            return (helpers.create_placeholder_image(),
                    f"Added ({x:.0f}, {y:.0f})",
                    s.homography.get_point_summary() if s else "")
        return (helpers.get_annotated_frame(s),
                f"Added ({x:.0f}, {y:.0f})",
                s.homography.get_point_summary())

    # ── Start / Finish Line ──────────────────────────────────────────────

    def _on_line_click(self, image, username, evt: gr.SelectData):
        """Handle click to set a start/finish-line endpoint."""
        s = self.app._get_session(username)
        if not s or not s.video_ready:
            return image, "Process video first", ""

        x, y = evt.index
        pts = s.start_finish_line

        if len(pts) >= 2:
            # Replace: start over
            s.start_finish_line = [[x, y]]
            status = f"Line point A set ({x:.0f}, {y:.0f}) — click point B"
        elif len(pts) == 1:
            s.start_finish_line.append([x, y])
            status = f"Line point B set ({x:.0f}, {y:.0f}) — line complete"
        else:
            s.start_finish_line = [[x, y]]
            status = f"Line point A set ({x:.0f}, {y:.0f}) — click point B"

        return (helpers.get_annotated_frame(s), status,
                s.homography.get_point_summary())

    def on_clear_line(self, username):
        """Clear the start/finish line."""
        s = self.app._get_session(username)
        if s:
            s.start_finish_line = []
            s.detected_laps = {}
        if not s or not s.video_ready:
            return helpers.create_placeholder_image(), "Line cleared", ""
        return helpers.get_annotated_frame(s), "Line cleared", ""

    # ── Compute Homography ──────────────────────────────────────────────

    def on_calculate_homo(self, r1w, r1h, r2w, r2h, r2dx, r2dy,
                          bev_w, bev_h, off_x, off_y, username):
        """Compute the homography matrix and return BEV preview."""
        s = self.app._get_session(username)
        if not s or not s.video_ready:
            return None, "Process video first"

        rect1 = RectangleSpec(width=r1w, height=r1h)
        rect2 = RectangleSpec(width=r2w, height=r2h,
                              offset_x=r2dx, offset_y=r2dy)

        success, msg = s.homography.compute(
            rect1, rect2, bev_w, bev_h, off_x, off_y
        )
        if not success:
            return None, msg

        # cv2.imread returns BGR — warp in BGR, convert to RGB at the end
        frame = cv2.imread(os.path.join(s.frames_dir, "00000.jpg"))
        warped = s.homography.warp_image(frame)

        bev_w_px, bev_h_px = s.homography.get_bev_dimensions()
        for i in range(0, bev_w_px, s.homography.resolution):
            cv2.line(warped, (i, 0), (i, bev_h_px), (100, 100, 100), 1)
        for i in range(0, bev_h_px, s.homography.resolution):
            cv2.line(warped, (0, i), (bev_w_px, i), (100, 100, 100), 1)

        info = f"BEV: {bev_w:.1f}x{bev_h:.1f}m"
        if off_x != 0 or off_y != 0:
            info += f" | Offset: ({off_x:+.1f}, {off_y:+.1f})m"
        cv2.putText(warped, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Create trajectory processor with actual video FPS
        s.traj_processor = TrajectoryProcessor(
            s.homography, video_fps=s.video_fps
        )

        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), msg

    # ── Save / Load ─────────────────────────────────────────────────────

    def on_save_homography(self, r1w, r1h, r2w, r2h, r2dx, r2dy,
                           bev_w, bev_h, off_x, off_y, username):
        """Save homography points, settings, and start/finish line to JSON."""
        s = self.app._get_session(username)
        if not s or s.homography.point_count < 4:
            return "Need at least 4 points to save"

        data = {
            "image_points": s.homography.image_points,
            "settings": {
                "rect1_width": r1w, "rect1_height": r1h,
                "rect2_width": r2w, "rect2_height": r2h,
                "rect2_offset_x": r2dx, "rect2_offset_y": r2dy,
                "bev_width": bev_w, "bev_height": bev_h,
                "offset_x": off_x, "offset_y": off_y,
            },
            "start_finish_line": s.start_finish_line,
        }

        save_path = os.path.join(s.exports_dir, "homography.json")
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        return f"Saved to {save_path}"

    def on_load_homography(self, username):
        """Load homography points, settings, and start/finish line from JSON."""
        s = self.app._get_session(username)
        defaults = (0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 15.0, 20.0, 0.0, 0.0)
        if not s:
            return (helpers.create_placeholder_image(),
                    "Not logged in", "", *defaults)

        load_path = os.path.join(s.exports_dir, "homography.json")
        if not os.path.exists(load_path):
            img = (helpers.get_annotated_frame(s) if s.video_ready
                   else helpers.create_placeholder_image())
            return (img, "No saved homography found", "", *defaults)

        with open(load_path, "r") as f:
            data = json.load(f)

        s.homography.clear_points()
        for pt in data.get("image_points", []):
            s.homography.add_point(pt[0], pt[1])

        # Restore start/finish line
        s.start_finish_line = data.get("start_finish_line", [])

        st = data.get("settings", {})

        line_msg = ""
        if s.start_finish_line and len(s.start_finish_line) == 2:
            line_msg = " + start/finish line"

        img = (helpers.get_annotated_frame(s) if s.video_ready
               else helpers.create_placeholder_image())
        return (
            img,
            f"Loaded {len(data.get('image_points', []))} points{line_msg}",
            s.homography.get_point_summary(),
            st.get("rect1_width", 0.5), st.get("rect1_height", 0.5),
            st.get("rect2_width", 0.5), st.get("rect2_height", 0.5),
            st.get("rect2_offset_x", 1.0), st.get("rect2_offset_y", 0.0),
            st.get("bev_width", 15.0), st.get("bev_height", 20.0),
            st.get("offset_x", 0.0), st.get("offset_y", 0.0),
        )
