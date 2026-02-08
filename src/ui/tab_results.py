"""
Results tab handlers — trajectory plots, lap data, G-force, animation (Tab 4).

Provides:
  - BEV trajectory overlay & coordinate plot
  - Speed / G-force profile chart
  - G-G scatter diagram
  - Lap time table with per-lap kinematics
  - Per-lap filtering for all visualisations
  - Combined tracking + BEV animation video
"""

import os
import logging
from typing import Optional, Dict

import cv2
import numpy as np
import gradio as gr
from PIL import Image

from ..core.lap_detector import LapDetector, filter_trajectories_by_lap
from .session import UserSession
from . import helpers

logger = logging.getLogger(__name__)


class ResultsHandlers:
    """Handlers for the Results tab."""

    def __init__(self, app):
        self.app = app

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_filtered_trajectories(self, s: UserSession):
        """Return raw trajectories filtered by the selected lap."""
        raw = s.tracker.trajectories
        if s.selected_lap == 0 or not s.detected_laps:
            return raw
        return filter_trajectories_by_lap(
            raw, s.detected_laps, s.selected_lap
        )

    def _filter_processed_by_lap(
        self, processed: Dict[int, np.ndarray], s: UserSession,
    ) -> Dict[int, np.ndarray]:
        """Slice *processed* arrays by the selected lap."""
        if s.selected_lap == 0 or not s.detected_laps:
            return processed
        filtered: Dict[int, np.ndarray] = {}
        for obj_id, data in processed.items():
            laps = s.detected_laps.get(obj_id, [])
            n = s.selected_lap
            if n <= len(laps):
                lap = laps[n - 1]
                filtered[obj_id] = data[
                    lap.traj_start_idx:lap.traj_end_idx + 1]
            else:
                filtered[obj_id] = data
        return filtered

    def _auto_detect_laps(self, s: UserSession):
        """Detect laps silently if a start/finish line exists."""
        if len(s.start_finish_line) != 2:
            return
        if not s.tracker or not s.tracker.trajectories:
            return
        detector = LapDetector(
            tuple(s.start_finish_line[0]),
            tuple(s.start_finish_line[1]),
        )
        s.detected_laps = {}
        for obj_id, traj in s.tracker.trajectories.items():
            s.detected_laps[obj_id] = detector.detect_laps(
                traj, video_fps=s.video_fps, min_lap_frames=30
            )

    # ── Data Export ─────────────────────────────────────────────────────

    def on_save_data(self, username):
        """Export trajectory data to JSON and CSV."""
        s = self.app._get_session(username)
        if not s or s.traj_processor is None:
            return "Homography not calculated"
        if s.tracker is None or not s.tracker.trajectories:
            return "No trajectories. Run tracking first."

        json_path = os.path.join(s.exports_dir, "trajectory_data.json")
        csv_path = os.path.join(s.exports_dir, "trajectory_data.csv")
        return s.traj_processor.export_data(
            s.tracker.trajectories, json_path, csv_path
        )

    # ── Lap Detection ──────────────────────────────────────────────────

    def on_detect_laps(self, username):
        """Detect laps and return summary text + dropdown choices."""
        s = self.app._get_session(username)
        if not s:
            return "Not logged in", gr.update(choices=["All"], value="All")
        if s.tracker is None or not s.tracker.trajectories:
            return ("No trajectories. Run tracking first.",
                    gr.update(choices=["All"], value="All"))
        if len(s.start_finish_line) != 2:
            return ("No start/finish line defined. "
                    "Draw one in the Calibration tab.",
                    gr.update(choices=["All"], value="All"))

        self._auto_detect_laps(s)

        summaries = []
        for obj_id, laps in s.detected_laps.items():
            if laps:
                summaries.append(f"— Object {obj_id} —")
                summaries.append(LapDetector.get_lap_summary(laps))
                summaries.append("")

        if not summaries:
            s.detected_laps = {}
            return ("No laps detected. Make sure the trajectory "
                    "crosses the start/finish line.",
                    gr.update(choices=["All"], value="All"))

        max_laps = max(
            (len(v) for v in s.detected_laps.values()), default=0)
        choices = ["All"] + [f"Lap {i + 1}" for i in range(max_laps)]
        s.selected_lap = 0

        return "\n".join(summaries), gr.update(choices=choices, value="All")

    def on_select_lap(self, lap_choice, username):
        """Store the selected lap number."""
        s = self.app._get_session(username)
        if not s:
            return
        if lap_choice == "All" or not lap_choice:
            s.selected_lap = 0
        else:
            try:
                s.selected_lap = int(lap_choice.split()[-1])
            except (ValueError, IndexError):
                s.selected_lap = 0

    # ── Generate All Plots ──────────────────────────────────────────────

    def on_plot(self, username):
        """
        Generate all result visualisations at once:
          coord plot, BEV overlay, speed profile, G-G diagram,
          statistics text, and lap table.
        """
        s = self.app._get_session(username)
        empty = (None, None, None, None, "", [])
        if not s or s.traj_processor is None:
            return (*empty[:4], "Homography not calculated", [])
        if s.tracker is None or not s.tracker.trajectories:
            return (*empty[:4], "No trajectories. Run tracking first.", [])

        # Auto-detect laps if line present
        if len(s.start_finish_line) == 2 and not s.detected_laps:
            self._auto_detect_laps(s)

        # ── Process full trajectory (for stats & lap table) ─────────
        full_processed = s.traj_processor.process_trajectories(
            s.tracker.trajectories)
        if not full_processed:
            return (*empty[:4], "No valid trajectory data", [])

        # ── Filtered view for plots ─────────────────────────────────
        display_processed = self._filter_processed_by_lap(
            full_processed, s)
        display_traj = self._get_filtered_trajectories(s)

        # ── 1. Coordinate trajectory plot ───────────────────────────
        coord_path = os.path.join(s.exports_dir, "trajectory_coords.png")
        coord_img, _ = s.traj_processor.visualize_matplotlib(
            display_traj, coord_path, "light")

        # ── 2. BEV overlay ──────────────────────────────────────────
        bev_img = self._generate_bev_overlay(s, display_traj)
        if bev_img is not None:
            Image.fromarray(bev_img).save(
                os.path.join(s.exports_dir, "trajectory_bev.png"))

        # ── 3. Speed / G-force profile ──────────────────────────────
        speed_path = os.path.join(s.exports_dir, "speed_profile.png")
        laps_for_plot = s.detected_laps if s.selected_lap == 0 else None
        speed_img = s.traj_processor.plot_speed_profile(
            display_processed, speed_path, laps_by_obj=laps_for_plot)

        # ── 4. G-G diagram ─────────────────────────────────────────
        gg_path = os.path.join(s.exports_dir, "gg_diagram.png")
        gg_img = s.traj_processor.plot_gforce_diagram(
            display_processed, gg_path)

        # ── 5. Extended stats text ──────────────────────────────────
        stats_text = s.traj_processor.format_extended_stats(
            display_processed)
        if s.selected_lap > 0:
            stats_text += f"\n[Showing Lap {s.selected_lap} only]"

        # ── 6. Lap table ────────────────────────────────────────────
        lap_rows: list = []
        if s.detected_laps and any(s.detected_laps.values()):
            _, lap_rows = s.traj_processor.compute_lap_table(
                full_processed, s.detected_laps)

        return coord_img, bev_img, speed_img, gg_img, stats_text, lap_rows

    # ── BEV overlay (internal) ─────────────────────────────────────────

    def _generate_bev_overlay(
        self, session: UserSession, trajectories=None
    ) -> Optional[np.ndarray]:
        """Trajectory overlaid on BEV-warped image. Returns RGB."""
        if session.homography.matrix is None:
            return None
        if trajectories is None:
            trajectories = session.tracker.trajectories

        frame = cv2.imread(os.path.join(session.frames_dir, "00000.jpg"))
        if frame is None:
            return None

        bev_w, bev_h = session.homography.get_bev_dimensions()
        bev_warped = session.homography.warp_image(frame)

        processed = session.traj_processor.process_trajectories(trajectories)
        if not processed:
            return cv2.cvtColor(bev_warped, cv2.COLOR_BGR2RGB)

        # Grid
        for i in range(0, bev_w, session.homography.resolution):
            cv2.line(bev_warped, (i, 0), (i, bev_h), (200, 200, 200), 1)
        for i in range(0, bev_h, session.homography.resolution):
            cv2.line(bev_warped, (0, i), (bev_w, i), (200, 200, 200), 1)

        # Start/finish line in BEV
        if (len(session.start_finish_line) == 2
                and session.homography.matrix is not None):
            sf_img = np.array(session.start_finish_line, dtype=np.float32)
            sf_bev = session.homography.transform_points(sf_img)
            if sf_bev is not None:
                p1 = tuple(sf_bev[0].astype(int))
                p2 = tuple(sf_bev[1].astype(int))
                cv2.line(bev_warped, p1, p2, (255, 100, 255), 3, cv2.LINE_AA)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(bev_warped, "S/F", (mid[0] - 15, mid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

        # Trajectory overlay — colour-code by accel/brake
        traj_layer = np.zeros_like(bev_warped)
        ACCEL  = (0, 255, 0)
        BRAKE  = (51, 51, 255)
        CONST  = (0, 255, 255)
        thresh = session.traj_processor.accel_threshold

        for obj_id, data in processed.items():
            pts    = data[:, 1:3].astype(np.int32)
            accels = data[:, 4]

            for i in range(len(pts) - 1):
                a = accels[i + 1]
                seg_color = ACCEL if a > thresh else (
                    BRAKE if a < -thresh else CONST)
                cv2.line(traj_layer, tuple(pts[i]), tuple(pts[i + 1]),
                         seg_color, 2, cv2.LINE_AA)

            if len(pts) > 0:
                cv2.circle(traj_layer, tuple(pts[0]), 5,
                           (0, 220, 0), -1, cv2.LINE_AA)
                cv2.circle(traj_layer, tuple(pts[0]), 6,
                           (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(traj_layer, tuple(pts[-1]), 5,
                           (0, 0, 220), -1, cv2.LINE_AA)
                cv2.circle(traj_layer, tuple(pts[-1]), 6,
                           (255, 255, 255), 1, cv2.LINE_AA)

        mask = (traj_layer.sum(axis=2) > 0).astype(np.uint8)[:, :, np.newaxis]
        bev_warped = np.where(
            mask,
            (bev_warped * 0.7 + traj_layer * 0.3).astype(np.uint8),
            bev_warped)

        # Scale bar
        px = session.homography.resolution
        cv2.line(bev_warped, (20, bev_h - 30), (20 + px, bev_h - 30),
                 (80, 80, 80), 2)
        cv2.putText(bev_warped, "1 m", (20, bev_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        # Legend
        y0 = 20
        for label, col in [("Accel", ACCEL), ("Brake", BRAKE),
                           ("Const", CONST)]:
            cv2.line(bev_warped, (bev_w - 120, y0),
                     (bev_w - 90, y0), col, 3)
            cv2.putText(bev_warped, label,
                        (bev_w - 85, y0 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            y0 += 22

        return cv2.cvtColor(bev_warped, cv2.COLOR_BGR2RGB)

    # ── Animation ───────────────────────────────────────────────────────

    def on_generate_animation(self, fps, trail_length, username,
                              progress=gr.Progress()):
        """Generate a combined tracking + BEV trajectory animation video."""
        s = self.app._get_session(username)
        if not s or s.traj_processor is None:
            return None, "Homography not calculated"
        if s.tracker is None or not s.tracker.trajectories:
            return None, "No trajectories. Run tracking first."

        tracked_frames = sorted(
            f for f in os.listdir(s.output_dir) if f.endswith('.png'))
        if not tracked_frames:
            return None, "No tracked frames. Run tracking first."

        progress(0, desc="Processing trajectories...")
        trajectories = self._get_filtered_trajectories(s)
        processed = s.traj_processor.process_trajectories(trajectories)
        if not processed:
            return None, "No valid trajectory points"

        frame_to_bev: dict = {}
        frame_to_img: dict = {}
        for obj_id, data in processed.items():
            for row in data:
                fidx = int(row[0])
                frame_to_bev.setdefault(fidx, {})[obj_id] = row[1:]
        for obj_id, traj in trajectories.items():
            for fidx, ix, iy in traj:
                frame_to_img.setdefault(fidx, {})[obj_id] = (ix, iy)

        progress(0.05, desc="Setting up video...")
        sample = None
        for f in tracked_frames:
            sample = cv2.imread(os.path.join(s.output_dir, f))
            if sample is not None:
                break
        if sample is None:
            return None, "Cannot read tracked frames"

        tracked_h, tracked_w = sample.shape[:2]
        bev_w_raw, bev_h_raw = s.homography.get_bev_dimensions()
        out_w, out_h = tracked_w, tracked_h * 2

        output_path = os.path.join(s.exports_dir,
                                   "trajectory_animation.mp4")
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (out_w, out_h))

        progress(0.1, desc="Rendering canvas...")
        traj_canvas = s.traj_processor.render_trajectory_canvas(
            processed, bev_w_raw, bev_h_raw)

        stats = s.traj_processor.get_stats(processed)
        y = 130
        for obj_id, st in stats.items():
            cv2.putText(
                traj_canvas,
                f"Obj{obj_id}: Top {st.top_speed:.1f} | "
                f"Avg {st.avg_speed:.1f} km/h",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

        colors_bgr = [(c[2], c[1], c[0]) for c in helpers.COLORS_RGB]
        WHITE = (255, 255, 255)
        ACCEL, BRAKE, CONST = (0, 255, 0), (51, 51, 255), (0, 255, 255)

        n_frames = len(tracked_frames)
        written, skipped = 0, 0

        for idx, frame_file in enumerate(tracked_frames):
            frame_idx = int(
                frame_file.replace('frame_', '').replace('.png', ''))

            tracked = cv2.imread(os.path.join(s.output_dir, frame_file))
            if tracked is None:
                skipped += 1
                continue
            if tracked.shape[:2] != (tracked_h, tracked_w):
                tracked = cv2.resize(tracked, (tracked_w, tracked_h))

            for obj_id, traj in trajectories.items():
                color = colors_bgr[(obj_id - 1) % len(colors_bgr)]
                past = [(ix, iy) for fi, ix, iy in traj
                        if fi <= frame_idx][-trail_length:]
                if len(past) > 1:
                    for i in range(len(past) - 1):
                        alpha = (i + 1) / len(past)
                        cv2.line(
                            tracked,
                            (int(past[i][0]), int(past[i][1])),
                            (int(past[i + 1][0]), int(past[i + 1][1])),
                            tuple(int(c * alpha) for c in color),
                            max(1, int(2 * alpha)), cv2.LINE_AA)
                if (frame_idx in frame_to_img
                        and obj_id in frame_to_img[frame_idx]):
                    cx, cy = map(int, frame_to_img[frame_idx][obj_id])
                    cv2.circle(tracked, (cx, cy), 5, color, -1, cv2.LINE_AA)
                    cv2.circle(tracked, (cx, cy), 7, WHITE, 1, cv2.LINE_AA)

            orig = cv2.imread(
                os.path.join(s.frames_dir, f"{frame_idx:05d}.jpg"))
            bev_warped = (
                s.homography.warp_image(orig) if orig is not None
                else np.full((bev_h_raw, bev_w_raw, 3), 35, dtype=np.uint8))
            bev_frame = cv2.addWeighted(
                bev_warped, 0.75, traj_canvas, 0.35, 0)

            for i in range(0, bev_w_raw, 50):
                cv2.line(bev_frame, (i, 0), (i, bev_h_raw),
                         (100, 100, 100), 1)
            for i in range(0, bev_h_raw, 50):
                cv2.line(bev_frame, (0, i), (bev_w_raw, i),
                         (100, 100, 100), 1)

            accel_thresh = s.traj_processor.accel_threshold
            if frame_idx in frame_to_bev:
                for obj_id, row in frame_to_bev[frame_idx].items():
                    bx, by, speed = float(row[0]), float(row[1]), float(row[2])
                    accel = float(row[3])
                    px, py = int(bx), int(by)
                    dot_color = (
                        ACCEL if accel > accel_thresh
                        else (BRAKE if accel < -accel_thresh else CONST))
                    cv2.circle(bev_frame, (px, py), 8,
                               dot_color, -1, cv2.LINE_AA)
                    cv2.circle(bev_frame, (px, py), 10,
                               WHITE, 2, cv2.LINE_AA)
                    cv2.putText(
                        bev_frame, f'{speed:.1f}', (px + 14, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

            bev_resized = cv2.resize(bev_frame, (out_w, tracked_h))
            combined = np.vstack([tracked, bev_resized])
            cv2.line(combined, (0, tracked_h), (out_w, tracked_h), WHITE, 2)
            out.write(combined)
            written += 1

            if idx % 25 == 0:
                progress(0.15 + 0.8 * idx / n_frames,
                         desc=f"Rendering: {idx}/{n_frames}")

        out.release()
        progress(1.0, desc="Done")

        status = f"{written} frames @ {fps}fps"
        if skipped:
            status += f" ({skipped} skipped)"
        status += "\nGreen=Accel, Red=Brake, Yellow=Const"
        if s.selected_lap > 0:
            status += f"\n[Showing Lap {s.selected_lap} only]"
        return output_path, status
