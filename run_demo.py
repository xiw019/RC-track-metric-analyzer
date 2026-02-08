#!/usr/bin/env python3
"""
SAM2 Video Demo - CLI Script

Processes a video file using Meta's Segment Anything Model 2.
For the full web UI, use: python main.py

Usage:
    # Interactive mode - click to select objects
    python run_demo.py --video demo.mov --interactive
    
    # Specify points directly (x,y coordinates)
    python run_demo.py --video demo.mov --points "100,200" "300,400"
    
    # Specify a bounding box (x1,y1,x2,y2)
    python run_demo.py --video demo.mov --box "100,100,400,300"
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.core.tracker import Tracker, TrackedObject


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Video Demo (CLI)")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for segmented frames",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=30,
        help="Save every N frames",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.model.model_id,
        choices=[
            "facebook/sam2.1-hiera-large",
            "facebook/sam2.1-hiera-base-plus",
            "facebook/sam2.1-hiera-small",
            "facebook/sam2.1-hiera-tiny",
        ],
        help="SAM2 model variant",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: click on the first frame to select objects",
    )
    parser.add_argument(
        "--points",
        type=str,
        nargs="+",
        help='Points to track as "x,y" (e.g., --points "100,200" "300,400")',
    )
    parser.add_argument(
        "--box",
        type=str,
        help='Bounding box as "x1,y1,x2,y2" (e.g., --box "100,100,400,300")',
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Frame index to add prompts on (default: 0)",
    )
    return parser.parse_args()


def interactive_select_points(frame_path):
    """Let user click on the image to select points to track."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("  LEFT CLICK:  Add point to track (green)")
    print("  RIGHT CLICK: Add negative point (red)")
    print("  MIDDLE/q:    Finish selection")
    print("  r:           Reset all points")
    print("  n:           Start new object")
    print("=" * 60)
    
    frame = np.array(Image.open(frame_path))
    
    objects = {1: TrackedObject(1)}
    current_obj_id = 1
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame)
    ax.set_title(f"Click to select objects (Object {current_obj_id})")
    ax.axis("off")
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    scatter_artists = []
    
    def update_plot():
        for artist in scatter_artists:
            artist.remove()
        scatter_artists.clear()
        
        for obj_id, obj in objects.items():
            if obj.is_empty:
                continue
            pts = np.array(obj.points)
            lbls = np.array(obj.labels)
            color = colors[(obj_id - 1) % 10]
            
            pos_mask = lbls == 1
            if pos_mask.any():
                s = ax.scatter(pts[pos_mask, 0], pts[pos_mask, 1],
                              c=[color], marker='o', s=200,
                              edgecolors='white', linewidths=2)
                scatter_artists.append(s)
            
            neg_mask = lbls == 0
            if neg_mask.any():
                s = ax.scatter(pts[neg_mask, 0], pts[neg_mask, 1],
                              c=[color], marker='x', s=200,
                              edgecolors='red', linewidths=3)
                scatter_artists.append(s)
        
        active = len([o for o in objects.values() if not o.is_empty])
        ax.set_title(f"Object {current_obj_id} | Total: {active}")
        fig.canvas.draw()
    
    def onclick(event):
        nonlocal current_obj_id
        if event.inaxes != ax:
            return
        
        x, y = event.xdata, event.ydata
        
        if event.button == 1:  # Left
            objects[current_obj_id].add_point(x, y, positive=True)
            print(f"  + Object {current_obj_id}: ({x:.0f}, {y:.0f})")
        elif event.button == 3:  # Right
            objects[current_obj_id].add_point(x, y, positive=False)
            print(f"  - Object {current_obj_id}: ({x:.0f}, {y:.0f})")
        elif event.button == 2:  # Middle
            plt.close()
            return
        
        update_plot()
    
    def onkey(event):
        nonlocal current_obj_id, objects
        
        if event.key == 'q':
            plt.close()
        elif event.key == 'r':
            objects = {1: TrackedObject(1)}
            current_obj_id = 1
            print("  Reset")
            update_plot()
        elif event.key == 'n':
            current_obj_id = max(objects.keys()) + 1
            objects[current_obj_id] = TrackedObject(current_obj_id)
            print(f"  New object: {current_obj_id}")
            update_plot()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    
    plt.tight_layout()
    plt.show()
    
    # Filter empty
    objects = {k: v for k, v in objects.items() if not v.is_empty}
    
    if not objects:
        print("No points selected!")
        return None
    
    print(f"\nSelected {len(objects)} object(s):")
    for obj_id, obj in objects.items():
        print(f"  Object {obj_id}: {obj.positive_count}+, {obj.negative_count}-")
    
    return objects


def parse_points(points_str_list):
    """Parse point strings like ['100,200', '300,400']."""
    points = []
    for p in points_str_list:
        x, y = map(float, p.split(','))
        points.append([x, y])
    return points


def show_mask(mask, ax, obj_id=None):
    """Overlay mask on matplotlib axis."""
    cmap = plt.cm.get_cmap("tab10")
    color = np.array([*cmap((obj_id - 1) % 10)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def main():
    args = parse_args()
    
    # Resolve paths
    video_path = args.video or config.paths.default_video
    output_dir = args.output or config.paths.output_dir
    frames_dir = config.paths.frames_dir
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SAM2 Video Demo (CLI)")
    print("=" * 60)
    print(f"Video:  {video_path}")
    print(f"Model:  {args.model}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Create tracker and extract frames
    tracker = Tracker(model_id=args.model)
    num_frames = tracker.extract_frames(video_path, frames_dir)
    
    # Get frame for prompts
    prompt_frame_path = os.path.join(frames_dir, f"{args.prompt_frame:05d}.jpg")
    first_frame = Image.open(prompt_frame_path)
    width, height = first_frame.size
    print(f"Dimensions: {width}x{height}, {num_frames} frames")
    
    # Collect prompts
    objects = {}
    
    if args.interactive:
        objects = interactive_select_points(prompt_frame_path)
        if not objects:
            print("No objects selected. Exiting.")
            sys.exit(0)
    elif args.box:
        coords = list(map(float, args.box.split(',')))
        if len(coords) != 4:
            print("Box must have 4 coords: x1,y1,x2,y2")
            sys.exit(1)
        # Box prompts need different handling - skip for simplicity
        print("Box prompts not implemented in CLI. Use --points or --interactive.")
        sys.exit(1)
    elif args.points:
        points = parse_points(args.points)
        obj = TrackedObject(1)
        for pt in points:
            obj.add_point(pt[0], pt[1], positive=True)
        objects = {1: obj}
        print(f"Tracking {len(points)} point(s)")
    else:
        # Default: center of frame
        obj = TrackedObject(1)
        obj.add_point(width / 2, height / 2, positive=True)
        objects = {1: obj}
        print(f"Tracking center: ({width/2:.0f}, {height/2:.0f})")
    
    # Run tracking
    print("\nRunning tracking...")
    
    def progress(p, msg):
        if p in (0.1, 0.3, 0.5, 0.8, 1.0):
            print(f"  [{p*100:.0f}%] {msg}")
    
    trajectories = tracker.track(objects, frames_dir, progress)
    
    # Save outputs
    print(f"\nSaving frames (every {args.frame_stride})...")
    saved = 0
    
    for frame_idx in range(0, num_frames, args.frame_stride):
        if frame_idx not in tracker.video_segments:
            continue
        
        frame = Tracker.load_frame(frames_dir, frame_idx)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(frame)
        axes[0].set_title(f"Frame {frame_idx}")
        axes[0].axis("off")
        
        axes[1].imshow(frame)
        for obj_id, mask in tracker.video_segments[frame_idx].items():
            show_mask(mask, axes[1], obj_id=obj_id)
        axes[1].set_title(f"Frame {frame_idx} - Tracked")
        axes[1].axis("off")
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        saved += 1
    
    print(f"\n{'=' * 60}")
    print(f"Done! Saved {saved} frames to: {output_dir}")
    
    # Print trajectory info
    print("\nTrajectories:")
    for obj_id, traj in trajectories.items():
        print(f"  Object {obj_id}: {len(traj)} points")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
