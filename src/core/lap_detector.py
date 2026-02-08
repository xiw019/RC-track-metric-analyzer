"""
Lap detection module — detects start/finish line crossings in trajectories.

Works in image-space: the user draws a line on the video frame, and crossings
are detected by checking trajectory segments against that line.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Lap:
    """Represents a single detected lap."""
    lap_number: int
    start_frame: int
    end_frame: int
    start_time: float       # seconds from video start
    end_time: float         # seconds from video start
    lap_time: float         # seconds
    traj_start_idx: int     # index into trajectory list
    traj_end_idx: int       # index into trajectory list


class LapDetector:
    """
    Detects laps by finding start/finish line crossings in a trajectory.

    The line is defined by two image-space points.  Crossings are detected
    using segment–segment intersection tests on the raw image-space trajectory.
    Only crossings in a consistent direction are counted to avoid
    back-and-forth noise.
    """

    def __init__(
        self,
        line_pt1: Tuple[float, float],
        line_pt2: Tuple[float, float],
    ):
        self.line_pt1 = np.array(line_pt1, dtype=np.float64)
        self.line_pt2 = np.array(line_pt2, dtype=np.float64)

    # ── Geometry helpers ─────────────────────────────────────────────────

    @staticmethod
    def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
        """2-D cross product (scalar)."""
        return float(a[0] * b[1] - a[1] * b[0])

    def _segments_intersect(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if trajectory segment (p1→p2) crosses the start/finish line.

        Returns:
            (intersects, cross_sign)  —  cross_sign encodes which side the
            trajectory exits on and is used to enforce uni-directional laps.
        """
        a, b = self.line_pt1, self.line_pt2
        d = b - a

        d1 = self._cross2d(d, p1 - a)
        d2 = self._cross2d(d, p2 - a)

        # Both on same side → no crossing
        if d1 * d2 > 0:
            return False, 0.0

        e = p2 - p1
        d3 = self._cross2d(e, a - p1)
        d4 = self._cross2d(e, b - p1)

        if d3 * d4 > 0:
            return False, 0.0

        return True, d2

    # ── Public API ───────────────────────────────────────────────────────

    def detect_crossings(
        self,
        trajectory: List[Tuple[int, float, float]],
    ) -> List[Tuple[int, int, float]]:
        """
        Detect every crossing of the start/finish line.

        Args:
            trajectory: [(frame_idx, x, y), …] in image space

        Returns:
            [(traj_index, frame_idx, cross_sign), …]
        """
        crossings: List[Tuple[int, int, float]] = []

        for i in range(len(trajectory) - 1):
            f1, x1, y1 = trajectory[i]
            f2, x2, y2 = trajectory[i + 1]

            ok, sign = self._segments_intersect(
                np.array([x1, y1]), np.array([x2, y2])
            )
            if ok:
                crossings.append((i + 1, f2, sign))

        return crossings

    def detect_laps(
        self,
        trajectory: List[Tuple[int, float, float]],
        video_fps: float = 30.0,
        min_lap_frames: int = 30,
    ) -> List[Lap]:
        """
        Detect laps from trajectory line-crossings.

        Only crossings in the *same* direction as the first crossing are
        counted, and a minimum frame gap is enforced to suppress noise.

        Args:
            trajectory:     [(frame_idx, x, y), …]
            video_fps:      Frames per second (for time computation)
            min_lap_frames: Minimum frame gap between valid crossings

        Returns:
            Ordered list of :class:`Lap` objects.
        """
        crossings = self.detect_crossings(trajectory)
        if not crossings:
            return []

        # Use first crossing's sign as reference direction
        ref_positive = crossings[0][2] > 0

        valid: List[Tuple[int, int, float]] = []
        for idx, frame, sign in crossings:
            if (sign > 0) != ref_positive:
                continue
            if valid and (frame - valid[-1][1]) < min_lap_frames:
                continue
            valid.append((idx, frame, sign))

        if len(valid) < 2:
            return []

        laps: List[Lap] = []
        for i in range(len(valid) - 1):
            s_idx, s_frame, _ = valid[i]
            e_idx, e_frame, _ = valid[i + 1]

            s_time = s_frame / video_fps
            e_time = e_frame / video_fps

            laps.append(Lap(
                lap_number=i + 1,
                start_frame=s_frame,
                end_frame=e_frame,
                start_time=s_time,
                end_time=e_time,
                lap_time=e_time - s_time,
                traj_start_idx=s_idx,
                traj_end_idx=e_idx,
            ))

        return laps

    # ── Formatting ───────────────────────────────────────────────────────

    @staticmethod
    def get_lap_summary(laps: List[Lap]) -> str:
        """Human-readable lap summary."""
        if not laps:
            return "No laps detected"

        best = min(laps, key=lambda l: l.lap_time)

        lines = [f"Laps detected: {len(laps)}", ""]
        for lap in laps:
            marker = "  ★ BEST" if lap is best else ""
            lines.append(
                f"  Lap {lap.lap_number}: {lap.lap_time:.3f}s "
                f"(frames {lap.start_frame}–{lap.end_frame}){marker}"
            )

        lines.append(f"\nBest Lap: #{best.lap_number} — {best.lap_time:.3f}s")

        if len(laps) > 1:
            avg = sum(l.lap_time for l in laps) / len(laps)
            lines.append(f"Average:  {avg:.3f}s")
            worst = max(laps, key=lambda l: l.lap_time)
            delta = worst.lap_time - best.lap_time
            lines.append(f"Spread:   {delta:.3f}s")

        return "\n".join(lines)


# ── Utility: filter trajectories by lap ─────────────────────────────────


def filter_trajectories_by_lap(
    raw_trajectories: Dict[int, List[Tuple[int, float, float]]],
    laps_by_obj: Dict[int, List[Lap]],
    lap_number: int,        # 0 = all laps
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Return a copy of *raw_trajectories* trimmed to the requested lap.

    If *lap_number* is 0 the trajectories are returned unchanged.
    """
    if lap_number == 0:
        return raw_trajectories

    filtered: Dict[int, List[Tuple[int, float, float]]] = {}
    for obj_id, traj in raw_trajectories.items():
        obj_laps = laps_by_obj.get(obj_id, [])
        if lap_number <= len(obj_laps):
            lap = obj_laps[lap_number - 1]
            filtered[obj_id] = traj[lap.traj_start_idx:lap.traj_end_idx + 1]
        else:
            # Object doesn't have this many laps — include everything
            filtered[obj_id] = traj
    return filtered
