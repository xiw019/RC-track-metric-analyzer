"""
Trajectory processing and visualization module.

Columns in processed arrays (from process_trajectories):
    0: frame_idx
    1: bev_x  (pixels)
    2: bev_y  (pixels)
    3: speed_kmh
    4: long_accel  (m/s² — longitudinal / tangential acceleration)
    5: lat_accel   (m/s² — lateral / centripetal acceleration)
    6: long_g      (longitudinal G-force)
    7: lat_g       (lateral G-force)
    8: cum_dist_m  (cumulative distance in metres)
"""

import os
import json
import csv
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..utils.drawing import smooth_trajectory, draw_circle, draw_polygon, draw_text


@dataclass
class TrajectoryStats:
    """Statistics for a trajectory."""
    obj_id: int
    num_points: int
    total_distance: float  # meters
    top_speed: float  # km/h
    avg_speed: float  # km/h


class TrajectoryProcessor:
    """
    Processes and visualizes object trajectories.

    Handles coordinate transformation, speed / G-force calculation,
    visualization, and data export.
    """

    ACCEL_COLOR = (0, 255, 0)   # Green - BGR
    BRAKE_COLOR = (51, 51, 255) # Red - BGR
    CONST_COLOR = (0, 255, 255) # Yellow - BGR
    WHITE = (255, 255, 255)

    COLORS_BGR = [
        (70, 57, 230), (157, 123, 69), (143, 157, 42),
        (74, 196, 233), (229, 93, 155), (97, 162, 244)
    ]

    def __init__(
        self,
        homography,  # HomographyTransform instance
        video_fps: float = 30.0,
        accel_threshold: float = 0.3,   # m/s² — gentle accel/brake detection
        smooth_window: int = 15         # Smoothing window for speed/accel
    ):
        self.homography = homography
        self.video_fps = video_fps
        self.accel_threshold = accel_threshold
        self.smooth_window = smooth_window

    # ── Smoothing ────────────────────────────────────────────────────────

    def _smooth_signal(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing to a signal."""
        if len(signal) < window:
            return signal
        kernel = np.ones(window) / window
        padded = np.pad(signal, (window // 2, window // 2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(signal)]

    # ── Core processing ──────────────────────────────────────────────────

    def process_trajectories(
        self,
        raw_trajectories: Dict[int, List[Tuple[int, float, float]]]
    ) -> Dict[int, np.ndarray]:
        """
        Process raw trajectories into BEV coordinates with full kinematics.

        Args:
            raw_trajectories: {obj_id: [(frame_idx, img_x, img_y), ...]}

        Returns:
            {obj_id: Nx9 array — see module docstring for column layout}
        """
        processed: Dict[int, np.ndarray] = {}

        for obj_id, traj in raw_trajectories.items():
            if not traj:
                continue

            N = len(traj)
            frame_idxs = np.array([t[0] for t in traj])
            img_pts = np.array([[t[1], t[2]] for t in traj], dtype=np.float32)

            bev_pts = self.homography.transform_points(img_pts)
            if bev_pts is None:
                continue

            # Smooth BEV coordinates
            if N > self.smooth_window:
                bev_pts[:, 0] = self._smooth_signal(
                    bev_pts[:, 0], self.smooth_window // 2)
                bev_pts[:, 1] = self._smooth_signal(
                    bev_pts[:, 1], self.smooth_window // 2)

            # ── Speed (scalar) ───────────────────────────────────────
            dt = np.diff(frame_idxs) / self.video_fps          # N-1
            dt[dt == 0] = 1 / self.video_fps

            dxy = np.diff(bev_pts, axis=0) / self.homography.resolution  # N-1, metres
            speeds_ms = np.linalg.norm(dxy, axis=1) / dt       # N-1
            speeds_kmh = np.concatenate([[0], speeds_ms * 3.6]) # N
            speeds_kmh = self._smooth_signal(speeds_kmh, self.smooth_window)

            # ── Longitudinal acceleration (tangential) ───────────────
            long_accel = np.concatenate(
                [[0], np.diff(speeds_kmh / 3.6) / dt])         # N, m/s²
            long_accel = self._smooth_signal(long_accel, self.smooth_window)

            # ── Lateral acceleration (centripetal) ───────────────────
            if len(dxy) >= 2:
                heading = np.arctan2(dxy[:, 1], dxy[:, 0])     # N-1
                heading_pad = np.concatenate([[heading[0]], heading])  # N
                d_heading = np.diff(heading_pad)                # N-1
                d_heading = np.arctan2(np.sin(d_heading),
                                       np.cos(d_heading))      # unwrap
                omega = np.concatenate([[0], d_heading / dt])   # N
                omega = self._smooth_signal(omega, self.smooth_window)
                lat_accel = (speeds_kmh / 3.6) * omega          # N, m/s²
            else:
                lat_accel = np.zeros(N)

            # ── G-forces ─────────────────────────────────────────────
            long_g = long_accel / 9.81
            lat_g  = lat_accel  / 9.81

            # ── Cumulative distance (metres) ─────────────────────────
            if len(dxy) > 0:
                cum_dist = np.concatenate(
                    [[0], np.cumsum(np.linalg.norm(dxy, axis=1))])  # N
            else:
                cum_dist = np.zeros(N)

            processed[obj_id] = np.column_stack([
                frame_idxs, bev_pts, speeds_kmh, long_accel,
                lat_accel, long_g, lat_g, cum_dist
            ])

        return processed

    # ── Statistics ────────────────────────────────────────────────────────

    def get_stats(
        self,
        processed: Dict[int, np.ndarray]
    ) -> Dict[int, TrajectoryStats]:
        """Calculate basic statistics for each trajectory."""
        stats = {}
        for obj_id, data in processed.items():
            speeds = data[:, 3]
            cum_dist = data[:, 8] if data.shape[1] > 8 else None

            if cum_dist is not None and len(cum_dist) > 0:
                total_dist = float(cum_dist[-1])
            elif len(data) > 1:
                bev_pts = data[:, 1:3]
                dists = np.linalg.norm(np.diff(bev_pts, axis=0), axis=1)
                total_dist = dists.sum() / self.homography.resolution
            else:
                total_dist = 0

            stats[obj_id] = TrajectoryStats(
                obj_id=obj_id,
                num_points=len(data),
                total_distance=total_dist,
                top_speed=float(speeds.max()),
                avg_speed=float(speeds.mean())
            )
        return stats

    def format_extended_stats(
        self,
        processed: Dict[int, np.ndarray]
    ) -> str:
        """Return a multi-line string with speed, distance and G-force info."""
        lines: List[str] = []
        for obj_id, data in processed.items():
            speeds   = data[:, 3]
            long_g   = data[:, 6]
            lat_g    = data[:, 7]
            cum_dist = data[:, 8]
            duration = (data[-1, 0] - data[0, 0]) / self.video_fps \
                       if len(data) > 1 else 0

            lines.append(f"— Object {obj_id} —")
            lines.append(
                f"  Distance: {cum_dist[-1]:.1f} m  |  "
                f"Duration: {duration:.1f} s")
            lines.append(
                f"  Top Speed: {speeds.max():.1f} km/h  |  "
                f"Avg: {speeds.mean():.1f} km/h")
            lines.append(
                f"  Max Accel: {long_g.max():.2f} G  |  "
                f"Max Brake: {abs(long_g.min()):.2f} G  |  "
                f"Max Lateral: {np.abs(lat_g).max():.2f} G")
            lines.append("")
        return "\n".join(lines)

    # ── Lap table ────────────────────────────────────────────────────────

    def compute_lap_table(
        self,
        processed: Dict[int, np.ndarray],
        laps_by_obj: Dict[int, list],
    ) -> Tuple[List[str], List[list]]:
        """
        Build a lap-time table with per-lap kinematics.

        Returns:
            (headers, rows)  — rows is a list-of-lists for gr.Dataframe.
        """
        headers = [
            "Obj", "Lap", "Time (s)", "Delta",
            "Top Speed (km/h)", "Avg Speed (km/h)",
            "Max Brake (G)", "Max Lat (G)",
        ]
        rows: List[list] = []

        for obj_id, laps in laps_by_obj.items():
            if not laps or obj_id not in processed:
                continue

            data = processed[obj_id]
            best_time = min(l.lap_time for l in laps)

            for lap in laps:
                s = lap.traj_start_idx
                e = min(lap.traj_end_idx + 1, len(data))
                chunk = data[s:e]
                if len(chunk) == 0:
                    continue

                spd   = chunk[:, 3]
                lg    = chunk[:, 6]
                latg  = chunk[:, 7]

                delta = lap.lap_time - best_time
                delta_str = "BEST" if delta < 0.001 else f"+{delta:.3f}"

                rows.append([
                    int(obj_id),
                    lap.lap_number,
                    f"{lap.lap_time:.3f}",
                    delta_str,
                    f"{spd.max():.1f}",
                    f"{spd.mean():.1f}",
                    f"{abs(lg.min()):.2f}" if lg.min() < 0 else "0.00",
                    f"{np.abs(latg).max():.2f}",
                ])

        return headers, rows

    # ── Matplotlib plots ─────────────────────────────────────────────────

    def plot_speed_profile(
        self,
        processed: Dict[int, np.ndarray],
        output_path: str,
        laps_by_obj: Optional[Dict[int, list]] = None,
    ) -> Optional[np.ndarray]:
        """
        Speed vs time (top) and longitudinal / lateral G vs time (bottom).

        Lap boundaries are drawn as vertical dashed lines when available.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image

        if not processed:
            return None

        fig, (ax_spd, ax_g) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]})
        fig.set_facecolor('#1a1a2e')
        ax_spd.set_facecolor('#16213e')
        ax_g.set_facecolor('#16213e')

        palette = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#9b5de5']

        for idx, (obj_id, data) in enumerate(processed.items()):
            color = palette[idx % len(palette)]
            t = data[:, 0] / self.video_fps

            ax_spd.plot(t, data[:, 3], color=color, linewidth=1.5,
                        label=f'Obj {obj_id}')

            ax_g.plot(t, data[:, 6], color=color, linewidth=1,
                      label=f'Obj {obj_id} Long G')
            ax_g.plot(t, np.abs(data[:, 7]), color=color, linewidth=1,
                      linestyle='--', alpha=0.6,
                      label=f'Obj {obj_id} |Lat G|')

            # Lap boundaries
            if laps_by_obj and obj_id in laps_by_obj:
                for lap in laps_by_obj[obj_id]:
                    lt = lap.start_frame / self.video_fps
                    ax_spd.axvline(x=lt, color='white', ls=':', alpha=0.35)
                    ax_g.axvline(x=lt, color='white', ls=':', alpha=0.35)
                    ax_spd.text(lt, ax_spd.get_ylim()[1] * 0.95,
                                f' L{lap.lap_number}',
                                color='white', fontsize=8, va='top')

        ax_spd.set_ylabel('Speed (km/h)', color='white', fontsize=11)
        ax_spd.legend(loc='upper right', fontsize=8, framealpha=0.5)
        ax_spd.grid(True, alpha=0.2, linestyle='--')
        ax_spd.tick_params(colors='white')
        ax_spd.set_title('Speed & G-Force Profile', color='white',
                         fontsize=14, fontweight='bold')

        ax_g.set_xlabel('Time (s)', color='white', fontsize=11)
        ax_g.set_ylabel('G-Force', color='white', fontsize=11)
        ax_g.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
        ax_g.legend(loc='upper right', fontsize=8, framealpha=0.5)
        ax_g.grid(True, alpha=0.2, linestyle='--')
        ax_g.tick_params(colors='white')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150,
                    facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        return np.array(Image.open(output_path))

    def plot_gforce_diagram(
        self,
        processed: Dict[int, np.ndarray],
        output_path: str,
    ) -> Optional[np.ndarray]:
        """
        G-G scatter diagram (lateral G vs longitudinal G).

        Points are colour-coded by speed.  Reference friction-circle
        rings are drawn for context.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image

        if not processed:
            return None

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')

        all_speeds = np.concatenate(
            [d[:, 3] for d in processed.values()])
        vmin, vmax = float(all_speeds.min()), float(all_speeds.max())

        last_sc = None
        for obj_id, data in processed.items():
            last_sc = ax.scatter(
                data[:, 7], data[:, 6],
                c=data[:, 3], cmap='plasma',
                vmin=vmin, vmax=vmax,
                s=4, alpha=0.45, label=f'Obj {obj_id}')

        # Friction-circle reference rings
        max_g = max(
            np.abs(np.concatenate([d[:, 6] for d in processed.values()])).max(),
            np.abs(np.concatenate([d[:, 7] for d in processed.values()])).max(),
            0.3,
        ) * 1.15
        for r in np.arange(0.25, max_g + 0.25, 0.25):
            circle = plt.Circle(
                (0, 0), r, fill=False, color='white',
                alpha=0.12, linestyle='--', linewidth=0.5)
            ax.add_patch(circle)
            if r < max_g:
                ax.text(r + 0.02, 0.02, f'{r:.2f}G',
                        color='white', fontsize=7, alpha=0.35)

        ax.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.3)
        ax.set_xlim(-max_g, max_g)
        ax.set_ylim(-max_g, max_g)
        ax.set_aspect('equal')
        ax.set_xlabel('Lateral G  (← Left  |  Right →)',
                       color='white', fontsize=11)
        ax.set_ylabel('Longitudinal G  (↓ Brake  |  Accel ↑)',
                       color='white', fontsize=11)
        ax.set_title('G-G Diagram', color='white',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.5)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.12, linestyle='--')

        if last_sc is not None:
            cbar = plt.colorbar(last_sc, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Speed (km/h)', color='white', fontsize=10)
            cbar.ax.tick_params(colors='white')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150,
                    facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        return np.array(Image.open(output_path))

    # ── BEV canvas rendering ─────────────────────────────────────────────

    def render_trajectory_canvas(
        self,
        processed: Dict[int, np.ndarray],
        width: int,
        height: int
    ) -> np.ndarray:
        """Render pre-computed trajectory lines on a canvas (BGR)."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for obj_id, data in processed.items():
            color = self.COLORS_BGR[(obj_id - 1) % len(self.COLORS_BGR)]
            pts = data[:, 1:3].astype(np.int32)
            accels = data[:, 4]

            for i in range(len(pts) - 1):
                accel = accels[i + 1]
                if accel > self.accel_threshold:
                    seg_color = self.ACCEL_COLOR
                elif accel < -self.accel_threshold:
                    seg_color = self.BRAKE_COLOR
                else:
                    seg_color = self.CONST_COLOR
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]),
                         seg_color, 2, cv2.LINE_AA)

            if len(pts) > 0:
                cv2.circle(canvas, tuple(pts[0]), 6,
                           self.WHITE, -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(pts[0]), 7, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(pts[-1]), 6,
                           color, -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(pts[-1]), 7,
                           self.WHITE, 1, cv2.LINE_AA)

        self._draw_legend(canvas)
        return canvas

    def _draw_legend(self, canvas: np.ndarray):
        cv2.putText(canvas, 'BEV Trajectory', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    self.WHITE, 2, cv2.LINE_AA)
        y = 50
        for label, color in [('Accel', self.ACCEL_COLOR),
                              ('Brake', self.BRAKE_COLOR),
                              ('Const', self.CONST_COLOR)]:
            cv2.line(canvas, (10, y + 10), (40, y + 10),
                     color, 4, cv2.LINE_AA)
            cv2.putText(canvas, label, (50, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.WHITE, 1, cv2.LINE_AA)
            y += 25

    # ── High-quality matplotlib trajectory plot ──────────────────────────

    def visualize_matplotlib(
        self,
        raw_trajectories: Dict[int, List[Tuple[int, float, float]]],
        output_path: str,
        style: str = "dark"
    ) -> Tuple[np.ndarray, str]:
        """Generate a high-quality BEV trajectory plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from PIL import Image

        all_coords = {}
        for obj_id, traj in raw_trajectories.items():
            if not traj:
                continue
            pts = np.array([[x, y] for _, x, y in traj], dtype=np.float32)
            bev_pts = self.homography.transform_points(pts)
            if bev_pts is not None:
                coords = bev_pts / self.homography.resolution
                coords[:, 1] = -coords[:, 1]
                if len(coords) > 5:
                    coords = smooth_trajectory(coords)
                all_coords[obj_id] = coords

        if not all_coords:
            return None, "No valid points after transformation"

        is_dark = style == "dark"
        plt.style.use('dark_background' if is_dark else 'seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(
            figsize=(12, 10),
            facecolor='#1a1a2e' if is_dark else 'white')
        if is_dark:
            ax.set_facecolor('#16213e')

        cmaps = ['plasma', 'viridis', 'cividis', 'cool', 'spring']
        colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#9b5de5']
        text_color = 'white' if is_dark else 'black'

        dist_info: List[str] = []
        for idx, (obj_id, coords) in enumerate(all_coords.items()):
            if len(coords) < 2:
                continue
            segments = np.concatenate(
                [coords[:-1, None], coords[1:, None]], axis=1)
            lc = LineCollection(
                segments, cmap=cmaps[idx % len(cmaps)],
                norm=plt.Normalize(0, len(coords) - 1), linewidth=2)
            lc.set_array(np.linspace(0, 1, len(coords) - 1))
            ax.add_collection(lc)

            color = colors[idx % len(colors)]
            ax.scatter(*coords[0], s=80, c='white', marker='o',
                       edgecolors=color, linewidths=2, zorder=5)
            ax.scatter(*coords[-1], s=80, c=color, marker='s',
                       edgecolors='white', linewidths=1.5, zorder=5)
            ax.annotate(
                f'Start {obj_id}', coords[0], xytext=(10, 10),
                textcoords='offset points', fontsize=10, color=text_color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=color, alpha=0.7))
            total_dist = np.sum(
                np.linalg.norm(np.diff(coords, axis=0), axis=1))
            dist_info.append(f"Obj {obj_id}: {total_dist:.2f}m")

        ax.autoscale()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X (meters)', fontsize=12, color=text_color)
        ax.set_ylabel('Y (meters)', fontsize=12, color=text_color)
        ax.set_title('Birds-Eye View Trajectory', fontsize=16,
                     fontweight='bold', color=text_color)
        ax.text(0.02, 0.98, '\n'.join(dist_info), transform=ax.transAxes,
                fontsize=11, verticalalignment='top', color=text_color,
                bbox=dict(boxstyle='round',
                          facecolor='#0f3460' if is_dark else '#f0f0f0',
                          alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150,
                    facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        img = np.array(Image.open(output_path))
        return img, f"Saved: {output_path}\n" + "\n".join(dist_info)

    # ── Data export ──────────────────────────────────────────────────────

    def export_data(
        self,
        raw_trajectories: Dict[int, List[Tuple[int, float, float]]],
        json_path: str,
        csv_path: str
    ) -> str:
        """Export trajectory data to JSON and CSV."""
        all_data: List[dict] = []
        data_by_obj: Dict[int, dict] = {}

        for obj_id, traj in raw_trajectories.items():
            if not traj:
                continue
            pts = np.array([[x, y] for _, x, y in traj], dtype=np.float32)
            bev_pts = self.homography.transform_points(pts)

            data_by_obj[obj_id] = {
                "object_id": obj_id,
                "num_points": len(traj),
                "points": []
            }
            for i, (frame_idx, x, y) in enumerate(traj):
                bev_x = float(bev_pts[i, 0]) if bev_pts is not None else None
                bev_y = float(bev_pts[i, 1]) if bev_pts is not None else None
                meters_x = bev_x / self.homography.resolution if bev_x else None
                meters_y = bev_y / self.homography.resolution if bev_y else None
                point = {
                    "frame_idx": frame_idx, "object_id": obj_id,
                    "image_x": float(x), "image_y": float(y),
                    "bev_x": bev_x, "bev_y": bev_y,
                    "meters_x": meters_x, "meters_y": meters_y,
                }
                all_data.append(point)
                data_by_obj[obj_id]["points"].append(point)

            if bev_pts is not None and len(bev_pts) > 1:
                dists = np.linalg.norm(np.diff(bev_pts, axis=0), axis=1)
                data_by_obj[obj_id]["total_distance_meters"] = float(
                    dists.sum() / self.homography.resolution)

        with open(json_path, 'w') as f:
            json.dump({
                "metadata": {
                    "num_objects": len(data_by_obj),
                    "total_points": len(all_data),
                    "bev_width_meters": self.homography.bev_width,
                    "bev_height_meters": self.homography.bev_height,
                    "pixels_per_meter": self.homography.resolution,
                },
                "objects": data_by_obj
            }, f, indent=2)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame_idx", "object_id", "image_x", "image_y",
                "bev_x", "bev_y", "meters_x", "meters_y"
            ])
            writer.writeheader()
            writer.writerows(all_data)

        lines = [f"JSON: {json_path}", f"CSV: {csv_path}",
                 f"Total: {len(all_data)} points"]
        for obj_id, data in data_by_obj.items():
            dist = data.get("total_distance_meters", 0)
            lines.append(f"Obj {obj_id}: {data['num_points']} pts, {dist:.2f}m")
        return "\n".join(lines)
