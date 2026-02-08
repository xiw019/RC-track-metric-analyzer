#!/usr/bin/env python3
"""
Track & Trajectory - Main Entry Point

Usage:
    python main.py                    # Launch Gradio web UI
    python main.py --video path.mp4   # Use custom video
    python main.py --port 8080        # Custom port
"""

import argparse
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.ui import create_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM2 Video Tracker with BEV Trajectory Visualization"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to input video (optional, can upload via UI)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config.ui.server_name,
        help=f"Server host (default: {config.ui.server_name})"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=config.ui.server_port,
        help=f"Server port (default: {config.ui.server_port})"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
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
        help=f"SAM2 model variant (default: {config.model.model_id})"
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    args = parse_args()
    
    # Determine video path (optional - can upload via UI)
    video_path = args.video
    if video_path and not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        video_path = None
    
    print("=" * 60)
    print("Track & Trajectory")
    print("=" * 60)
    if video_path:
        print(f"Video:  {video_path}")
    else:
        print("Video:  (upload via UI)")
    print(f"Model:  {args.model}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nUI will start. Upload video or click 'Process Video' to begin.")
    print("=" * 60)
    
    # Create and launch app (initialization is user-triggered)
    app = create_app(
        frames_dir=config.paths.frames_dir,
        output_dir=config.paths.output_dir,
        exports_dir=config.paths.exports_dir,
        video_path=video_path,
        model_id=args.model
    )
    
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
