"""
Shared UI constants and helper functions.

Used by all tab handler modules and the main app layout.
"""

import os
import logging
import cv2
import numpy as np

from ..core.tracker import Tracker
from ..utils.drawing import draw_circle, draw_polygon

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

COLORS_RGB = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]

MAX_VIDEO_SIZE_MB = 50

# ── Placeholder / Frame Loading ────────────────────────────────────────


def create_placeholder_image(text: str = "Process video first") -> np.ndarray:
    """Create a dark placeholder image with centered text."""
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.putText(img, text, (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (200, 200, 200), 2)
    return img


def load_frame(session, idx: int = 0) -> np.ndarray:
    """Load a video frame as RGB. Returns placeholder when not ready."""
    if not session.video_ready:
        return create_placeholder_image()
    return Tracker.load_frame(session.frames_dir, idx)


# ── Annotation Drawing ─────────────────────────────────────────────────


def draw_frame_annotations(session, frame: np.ndarray) -> np.ndarray:
    """Draw tracking points and homography reference on *frame* (RGB)."""
    result = frame.copy()

    # Tracking points
    for obj_id, obj in session.objects.items():
        color = COLORS_RGB[(obj_id - 1) % len(COLORS_RGB)]
        for pt, lbl in zip(obj.points, obj.labels):
            x, y = int(pt[0]), int(pt[1])
            if lbl == 1:
                draw_circle(result, (x, y), 12, color, -1, (255, 255, 255))
            else:
                # Negative point: red X
                cv2.line(result, (x - 10, y - 10), (x + 10, y + 10),
                         (255, 0, 0), 3)
                cv2.line(result, (x - 10, y + 10), (x + 10, y - 10),
                         (255, 0, 0), 3)
            label = f"Obj{obj_id}" + ("" if lbl == 1 else " (neg)")
            cv2.putText(result, label, (x + 15, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Homography reference points
    if session.homography.image_points:
        for i, (x, y) in enumerate(session.homography.image_points):
            draw_circle(result, (x, y), 10, (0, 255, 255), -1, (0, 0, 0))
            label = f"R{1 if i < 4 else 2}-P{(i % 4) + 1}"
            cv2.putText(result, label, (int(x) + 15, int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(session.homography.image_points) >= 4:
            draw_polygon(result, session.homography.image_points[:4],
                         (0, 255, 255))
        if len(session.homography.image_points) >= 8:
            draw_polygon(result, session.homography.image_points[4:8],
                         (255, 200, 0))

    # Start / Finish line
    _draw_start_finish_line(result, session)

    return result


def _draw_start_finish_line(frame: np.ndarray, session) -> None:
    """Draw the start/finish line and its endpoints on *frame* (RGB)."""
    pts = session.start_finish_line
    if not pts:
        return

    LINE_COLOR = (255, 100, 255)   # magenta-ish
    LABEL_COLOR = (255, 100, 255)

    for i, (x, y) in enumerate(pts):
        draw_circle(frame, (x, y), 10, LINE_COLOR, -1, (255, 255, 255))
        label = f"SF-{'A' if i == 0 else 'B'}"
        cv2.putText(frame, label, (int(x) + 15, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_COLOR, 2)

    if len(pts) == 2:
        p1 = (int(pts[0][0]), int(pts[0][1]))
        p2 = (int(pts[1][0]), int(pts[1][1]))
        cv2.line(frame, p1, p2, LINE_COLOR, 3, cv2.LINE_AA)
        # Dashed cross-hatching for visibility
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(frame, "START/FINISH", (mid_x - 60, mid_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def get_annotated_frame(session) -> np.ndarray:
    """Get the prompt frame with all annotations overlaid."""
    return draw_frame_annotations(
        session, load_frame(session, session.prompt_frame_idx)
    )


# ── Points Summary ─────────────────────────────────────────────────────


def get_points_summary(session) -> str:
    """Return a human-readable summary of selected tracking points."""
    if not session.objects:
        return "No points selected"
    lines = []
    for obj_id, obj in session.objects.items():
        if not obj.is_empty:
            lines.append(
                f"Obj {obj_id}: {obj.positive_count}+, {obj.negative_count}-"
            )
    return "\n".join(lines) or "No points selected"


# ── Cache Cleanup ──────────────────────────────────────────────────────


def clear_frames(session) -> None:
    """Clear all extracted JPEG frames from the session cache."""
    if os.path.exists(session.frames_dir):
        for f in os.listdir(session.frames_dir):
            if f.endswith('.jpg'):
                try:
                    os.remove(os.path.join(session.frames_dir, f))
                except OSError as e:
                    logger.warning("Failed to remove frame %s: %s", f, e)


def clear_output(session) -> None:
    """Clear all output PNG frames from the session cache."""
    if os.path.exists(session.output_dir):
        for f in os.listdir(session.output_dir):
            if f.endswith('.png'):
                try:
                    os.remove(os.path.join(session.output_dir, f))
                except OSError as e:
                    logger.warning("Failed to remove output %s: %s", f, e)
