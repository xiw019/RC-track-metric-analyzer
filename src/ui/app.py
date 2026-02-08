"""
Gradio Web UI for Track & Trajectory.

This is the main orchestrator that:
  - Composes per-tab handler objects (video, tracking, homography, results)
  - Manages per-user sessions
  - Defines the Gradio layout and event bindings
"""

import os
import logging
from typing import Optional

import gradio as gr

from ..auth import UserManager
from .session import UserSession
from . import helpers
from .tab_video import VideoHandlers
from .tab_tracking import TrackingHandlers
from .tab_homography import HomographyHandlers
from .tab_results import ResultsHandlers

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
#  AppUI — thin coordinator
# ════════════════════════════════════════════════════════════════════════

class AppUI:
    """
    Gradio-based web interface for SAM2 tracking and BEV trajectory.

    All per-user mutable state lives in UserSession objects (keyed by
    username).  The SAM2 model itself is shared across sessions via
    Tracker's class-level cache.
    """

    def __init__(self, frames_dir: str, output_dir: str, exports_dir: str,
                 model_id: str = "facebook/sam2.1-hiera-tiny"):
        self._default_frames_dir = frames_dir
        self._default_output_dir = output_dir
        self._default_exports_dir = exports_dir
        self.model_id = model_id
        self.root_dir = os.path.dirname(os.path.dirname(frames_dir))
        self._sessions: dict[str, UserSession] = {}

        for d in [frames_dir, output_dir, exports_dir]:
            os.makedirs(d, exist_ok=True)

        # Per-tab handler groups
        self.video = VideoHandlers(self)
        self.tracking = TrackingHandlers(self)
        self.homography = HomographyHandlers(self)
        self.results = ResultsHandlers(self)

    # ── Session management ──────────────────────────────────────────────

    def _get_session(self, username: str) -> Optional[UserSession]:
        """Get or create a session for the given user."""
        if not username:
            return None
        if username not in self._sessions:
            user_dir = os.path.join(self.root_dir, "data", "users", username)
            session = UserSession(
                username=username,
                frames_dir=os.path.join(user_dir, "cache", "frames"),
                output_dir=os.path.join(user_dir, "cache", "output"),
                exports_dir=os.path.join(user_dir, "cache", "exports"),
                model_id=self.model_id,
            )
            os.makedirs(os.path.join(user_dir, "videos"), exist_ok=True)
            self._sessions[username] = session
            logger.info("Created session for user: %s", username)
        return self._sessions[username]

    def _get_user_videos_dir(self, username: str) -> str:
        return os.path.join(self.root_dir, "data", "users", username, "videos")

    # ── Helpers for .then() chains ──────────────────────────────────────

    def _get_track_image(self, username):
        s = self._get_session(username)
        if s and s.video_ready:
            return helpers.get_annotated_frame(s)
        return helpers.create_placeholder_image()

    def _is_video_ready(self, username):
        s = self._get_session(username)
        return gr.update(interactive=bool(s and s.video_ready))


# ════════════════════════════════════════════════════════════════════════
#  create_app — Gradio layout + event wiring
# ════════════════════════════════════════════════════════════════════════

def create_app(
    frames_dir: str,
    output_dir: str,
    exports_dir: str,
    video_path: Optional[str] = None,
    **kwargs,
) -> gr.Blocks:
    """Create and return the complete Gradio application."""
    ui = AppUI(frames_dir, output_dir, exports_dir, **kwargs)
    user_manager = UserManager()

    custom_css = """
    .login-header { text-align: center; margin-bottom: 2rem; }
    .welcome-banner { background: #2d3748; color: white; padding: 0.75rem 1rem;
                      border-radius: 6px; margin-bottom: 0.5rem; }
    """

    with gr.Blocks(title="Track & Trajectory", css=custom_css) as app:
        logged_in_user = gr.State(value=None)

        # ─────────────── Login Section ──────────────────────────────────
        with gr.Column(visible=True) as login_section:
            gr.Markdown(
                """
                <div class="login-header">
                    <h1>Track & Trajectory</h1>
                    <p>SAM2 Video Tracker with BEV Trajectory</p>
                </div>
                """
            )

            with gr.Tabs() as auth_tabs:
                with gr.TabItem("Login"):
                    login_username = gr.Textbox(
                        label="Username", placeholder="Enter username",
                        max_lines=1,
                    )
                    login_password = gr.Textbox(
                        label="Password", placeholder="Enter password",
                        type="password", max_lines=1,
                    )
                    login_btn = gr.Button("Login", variant="primary",
                                          size="lg")
                    login_status = gr.Markdown("")

                with gr.TabItem("Register"):
                    reg_username = gr.Textbox(
                        label="Username", placeholder="3-20 characters",
                        max_lines=1,
                    )
                    reg_password = gr.Textbox(
                        label="Password", placeholder="Min 8 characters",
                        type="password", max_lines=1,
                    )
                    reg_password_confirm = gr.Textbox(
                        label="Confirm Password", type="password",
                        max_lines=1,
                    )
                    register_btn = gr.Button("Register", variant="primary",
                                             size="lg")
                    register_status = gr.Markdown("")

            user_count_info = gr.Markdown(
                f"*{user_manager.get_user_count()} registered users*"
            )

        # ─────────────── Main Application ───────────────────────────────
        with gr.Column(visible=False) as main_section:
            with gr.Row():
                welcome_msg = gr.Markdown("",
                                          elem_classes=["welcome-banner"])
                logout_btn = gr.Button("Logout", size="sm", scale=0)

            gr.Markdown("# Track & Trajectory")

            with gr.Tabs() as main_tabs:

                # ==================== Tab 1: Video ======================
                with gr.TabItem("1. Video"):
                    gr.Markdown(
                        "**Step 1: Upload and process your video**\n\n"
                        "Upload a video file (max 50 MB). The system will "
                        "extract frames for analysis. Your videos are saved "
                        "and available when you log back in."
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            video_preview = gr.Video(
                                label="Upload New Video",
                                value=video_path, height=250,
                            )
                            with gr.Row():
                                saved_videos_dropdown = gr.Dropdown(
                                    label="Or load a previous video",
                                    choices=[], value=None, interactive=True,
                                )
                                load_video_btn = gr.Button("Load", size="sm")
                        with gr.Column(scale=1):
                            upload_status = gr.Textbox(
                                label="Status",
                                value=(
                                    "Upload a video to start"
                                    if video_path is None
                                    else f"Ready: {os.path.basename(video_path)}"
                                ),
                                lines=2,
                            )
                            process_btn = gr.Button(
                                "Process Video", variant="primary", size="lg",
                                interactive=video_path is not None,
                            )

                    process_status = gr.Textbox(
                        label="Progress",
                        value="Upload video, then click Process",
                    )
                    frame_preview = gr.Image(
                        value=helpers.create_placeholder_image(
                            "Upload and process video"
                        ),
                        label="First Frame Preview", height=400,
                    )

                # ==================== Tab 2: Calibration ================
                with gr.TabItem("2. Calibration"):
                    gr.Markdown(
                        "**Step 2: Set up camera calibration (Homography) "
                        "& Start/Finish Line**"
                        "\n\n"
                        "Calibration transforms the camera view to a bird's-"
                        "eye view (BEV) for accurate distance measurements."
                        "\n\n"
                        "**Homography — How it works:**\n"
                        "1. Select *Homography Points* mode\n"
                        "2. Click 4 corners of a known rectangle in the "
                        "video (TL → TR → BR → BL)\n"
                        "3. Enter the real-world dimensions in meters\n"
                        "4. Click Calculate\n\n"
                        "**Start/Finish Line:**\n"
                        "1. Switch to *Start/Finish Line* mode\n"
                        "2. Click two points to define the line\n"
                        "3. Use *Detect Laps* in the Results tab to get "
                        "lap times"
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            calib_mode = gr.Radio(
                                ["Homography Points", "Start/Finish Line"],
                                value="Homography Points",
                                label="Click Mode",
                            )
                            homo_img = gr.Image(
                                value=helpers.create_placeholder_image(
                                    "Process video first"
                                ),
                                label="Click corners (TL, TR, BR, BL)",
                                interactive=True,
                            )
                            with gr.Row():
                                homo_undo = gr.Button("Undo")
                                homo_clear = gr.Button("Clear",
                                                       variant="stop")
                                line_clear = gr.Button(
                                    "Clear Line", variant="secondary",
                                )

                        with gr.Column(scale=1):
                            homo_status = gr.Textbox(label="Status")
                            homo_info = gr.Textbox(label="Points", lines=6)

                            with gr.Row():
                                mx = gr.Number(label="X", value=0, scale=1)
                                my = gr.Number(label="Y", value=0, scale=1)
                                add_btn = gr.Button("+", scale=0)

                            gr.Markdown("**Rectangle 1 (meters)**")
                            with gr.Row():
                                r1w = gr.Number(label="Width", value=0.5)
                                r1h = gr.Number(label="Height", value=0.5)

                            gr.Markdown("**Rectangle 2 (optional)**")
                            with gr.Row():
                                r2w = gr.Number(label="Width", value=0.5)
                                r2h = gr.Number(label="Height", value=0.5)
                            with gr.Row():
                                r2dx = gr.Number(label="Offset X", value=1.0)
                                r2dy = gr.Number(label="Offset Y", value=0.0)

                            gr.Markdown("**BEV Output (meters)**")
                            with gr.Row():
                                bw = gr.Number(label="Width", value=15.0)
                                bh = gr.Number(label="Height", value=20.0)

                            gr.Markdown("**BEV Offset**")
                            with gr.Row():
                                off_x = gr.Number(label="X", value=0.0)
                                off_y = gr.Number(label="Y", value=0.0)

                            calc_btn = gr.Button(
                                "Calculate", variant="primary", size="lg",
                                interactive=False,
                            )

                            with gr.Row():
                                homo_save = gr.Button("Save", size="sm")
                                homo_load = gr.Button("Load", size="sm")
                            homo_io_status = gr.Textbox(
                                label="", lines=1, show_label=False,
                            )

                    homo_preview = gr.Image(label="BEV Preview", height=350)
                    homo_result = gr.Textbox(label="Result")

                # ==================== Tab 3: Tracking ===================
                with gr.TabItem("3. Tracking"):
                    gr.Markdown(
                        "**Step 3: Select objects and run SAM2 tracking**"
                        "\n\n"
                        "Click on objects you want to track. SAM2 will "
                        "automatically segment and track them.\n\n"
                        "**Instructions:**\n"
                        "1. Use the frame slider to find the best frame\n"
                        "2. Select 'Add Point' or 'Exclude Point' mode\n"
                        "3. Click on the object\n"
                        "4. Click 'New Object' for additional objects\n"
                        "5. Click 'Run Tracking' to start"
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_frame_slider = gr.Slider(
                                0, 0, 0, step=1,
                                label="Prompt Frame (select which frame "
                                      "to mark objects on)",
                                interactive=False,
                            )
                            track_img = gr.Image(
                                value=helpers.create_placeholder_image(
                                    "Process video first"
                                ),
                                label="Click to select objects",
                                interactive=True, height=450,
                            )
                            with gr.Row():
                                point_mode = gr.Radio(
                                    ["Add Point", "Exclude Point"],
                                    value="Add Point", label="Click Mode",
                                )
                            with gr.Row():
                                new_btn = gr.Button(
                                    "New Object", variant="primary",
                                    interactive=False,
                                )
                                undo_btn = gr.Button("Undo")
                                clear_btn = gr.Button("Clear",
                                                      variant="stop")

                        with gr.Column(scale=1):
                            track_status = gr.Textbox(label="Status")
                            points_info = gr.Textbox(
                                label="Selected Points", lines=5, value="",
                            )
                            stride = gr.Slider(
                                1, 60, 1, step=1,
                                label="Save every N frames",
                            )
                            run_btn = gr.Button(
                                "Run Tracking", variant="primary", size="lg",
                                interactive=False,
                            )

                    gallery = gr.Gallery(
                        label="Sample Results", columns=4, height=200,
                    )
                    result_txt = gr.Textbox(label="Result")

                    gr.Markdown("**Export Video**")
                    with gr.Row():
                        video_fps = gr.Slider(
                            10, 60, 30, step=5, label="FPS",
                        )
                        video_btn = gr.Button("Generate Video",
                                              interactive=False)
                    video_out = gr.Video(label="Output Video")

                # ==================== Tab 4: Results ====================
                with gr.TabItem("4. Results"):
                    gr.Markdown(
                        "**Step 4: View and export trajectory results**\n\n"
                        "Generate plots, view lap data, analyse speed & "
                        "G-force, and create animations."
                    )

                    # ── Actions ─────────────────────────────────
                    with gr.Row():
                        save_btn = gr.Button("Save Data (JSON/CSV)")
                        detect_laps_btn = gr.Button(
                            "Detect Laps", variant="secondary")
                        plot_btn = gr.Button("Generate All Plots",
                                             variant="primary")
                        save_status = gr.Textbox(
                            label="Status", scale=2, lines=1)

                    # ── Lap selector ────────────────────────────
                    with gr.Row():
                        lap_selector = gr.Dropdown(
                            label="View Lap",
                            choices=["All"], value="All",
                            interactive=True, scale=1,
                        )
                        lap_summary = gr.Textbox(
                            label="Lap Summary", lines=4, scale=3)

                    # ── Lap time table ──────────────────────────
                    with gr.Accordion("Lap Times", open=True):
                        lap_table = gr.Dataframe(
                            headers=[
                                "Obj", "Lap", "Time (s)", "Delta",
                                "Top Speed (km/h)", "Avg Speed (km/h)",
                                "Max Brake (G)", "Max Lat (G)",
                            ],
                            label="Lap Data",
                            interactive=False,
                            wrap=True,
                        )

                    # ── Trajectory plots ────────────────────────
                    with gr.Accordion("Trajectory", open=True):
                        with gr.Row():
                            bev_img = gr.Image(
                                label="BEV Overlay", height=420)
                            coord_img = gr.Image(
                                label="Coordinates", height=420)

                    # ── Speed & G-force ─────────────────────────
                    with gr.Accordion("Speed & G-Force", open=True):
                        with gr.Row():
                            speed_img = gr.Image(
                                label="Speed / G-Force Profile",
                                height=380)
                            gg_img = gr.Image(
                                label="G-G Diagram", height=380)

                    # ── Statistics ──────────────────────────────
                    plot_info = gr.Textbox(
                        label="Statistics", lines=6)

                    # ── Animation ───────────────────────────────
                    with gr.Accordion("Animation", open=False):
                        with gr.Row():
                            anim_fps = gr.Slider(
                                10, 60, 30, step=5, label="FPS")
                            anim_trail = gr.Slider(
                                10, 200, 50, step=10,
                                label="Trail Length")
                            anim_btn = gr.Button(
                                "Generate Animation",
                                variant="primary")
                        with gr.Row():
                            anim_video = gr.Video(label="Animation")
                            anim_status = gr.Textbox(label="Status")

        # ═══════════════ Event Bindings ═════════════════════════════════

        # -- Video tab ---------------------------------------------------
        video_preview.change(
            ui.video.on_video_upload,
            inputs=[video_preview, logged_in_user],
            outputs=[upload_status, process_btn, saved_videos_dropdown],
        )
        load_video_btn.click(
            lambda name, user: ui.video.load_user_video(name, user),
            inputs=[saved_videos_dropdown, logged_in_user],
            outputs=[video_preview, upload_status, process_btn],
        )
        process_btn.click(
            ui.video.on_process_video,
            inputs=[logged_in_user],
            outputs=[frame_preview, process_status, calc_btn,
                     homo_img, new_btn, prompt_frame_slider],
        ).then(
            ui._get_track_image,
            inputs=[logged_in_user],
            outputs=[track_img],
        ).then(
            ui._is_video_ready,
            inputs=[logged_in_user],
            outputs=[run_btn],
        ).then(
            ui._is_video_ready,
            inputs=[logged_in_user],
            outputs=[video_btn],
        )

        # -- Frame selection ---------------------------------------------
        prompt_frame_slider.change(
            ui.video.on_frame_change,
            inputs=[prompt_frame_slider, logged_in_user],
            outputs=[track_img],
        )

        # -- Calibration tab ---------------------------------------------
        homo_img.select(
            ui.homography.on_homo_click,
            inputs=[homo_img, calib_mode, logged_in_user],
            outputs=[homo_img, homo_status, homo_info],
        )
        homo_clear.click(
            ui.homography.on_clear_homo,
            inputs=[logged_in_user],
            outputs=[homo_img, homo_status, homo_info],
        )
        homo_undo.click(
            ui.homography.on_undo_homo,
            inputs=[logged_in_user],
            outputs=[homo_img, homo_status, homo_info],
        )
        line_clear.click(
            ui.homography.on_clear_line,
            inputs=[logged_in_user],
            outputs=[homo_img, homo_status, homo_info],
        )
        add_btn.click(
            ui.homography.on_add_manual_homo,
            inputs=[mx, my, logged_in_user],
            outputs=[homo_img, homo_status, homo_info],
        )
        calc_btn.click(
            ui.homography.on_calculate_homo,
            inputs=[r1w, r1h, r2w, r2h, r2dx, r2dy,
                    bw, bh, off_x, off_y, logged_in_user],
            outputs=[homo_preview, homo_result],
        )
        homo_save.click(
            ui.homography.on_save_homography,
            inputs=[r1w, r1h, r2w, r2h, r2dx, r2dy,
                    bw, bh, off_x, off_y, logged_in_user],
            outputs=[homo_io_status],
        )
        homo_load.click(
            ui.homography.on_load_homography,
            inputs=[logged_in_user],
            outputs=[homo_img, homo_status, homo_info,
                     r1w, r1h, r2w, r2h, r2dx, r2dy, bw, bh, off_x, off_y],
        )

        # -- Tracking tab ------------------------------------------------
        track_img.select(
            ui.tracking.on_track_click,
            inputs=[track_img, point_mode, logged_in_user],
            outputs=[track_img, track_status, points_info],
        )
        new_btn.click(
            ui.tracking.on_new_object,
            inputs=[logged_in_user],
            outputs=[track_img, track_status, points_info],
        )
        undo_btn.click(
            ui.tracking.on_undo_point,
            inputs=[logged_in_user],
            outputs=[track_img, track_status, points_info],
        )
        clear_btn.click(
            ui.tracking.on_clear_points,
            inputs=[logged_in_user],
            outputs=[track_img, track_status, points_info],
        )
        run_btn.click(
            ui.tracking.on_run_tracking,
            inputs=[stride, prompt_frame_slider, logged_in_user],
            outputs=[gallery, result_txt],
        )
        video_btn.click(
            ui.tracking.on_generate_video,
            inputs=[video_fps, logged_in_user],
            outputs=[video_out, result_txt],
        )

        # -- Results tab -------------------------------------------------
        save_btn.click(
            ui.results.on_save_data,
            inputs=[logged_in_user],
            outputs=[save_status],
        )
        detect_laps_btn.click(
            ui.results.on_detect_laps,
            inputs=[logged_in_user],
            outputs=[lap_summary, lap_selector],
        )
        lap_selector.change(
            ui.results.on_select_lap,
            inputs=[lap_selector, logged_in_user],
        )
        plot_btn.click(
            ui.results.on_plot,
            inputs=[logged_in_user],
            outputs=[coord_img, bev_img, speed_img, gg_img,
                     plot_info, lap_table],
        )
        anim_btn.click(
            ui.results.on_generate_animation,
            inputs=[anim_fps, anim_trail, logged_in_user],
            outputs=[anim_video, anim_status],
        )

        # ═══════════════ Auth Handlers ══════════════════════════════════

        def do_login(username, password):
            success, message, user = user_manager.login(username, password)
            if success:
                ui._get_session(username)   # ensure session exists
                saved_videos = ui.video.get_user_videos(username)
                video_choices = [v["name"] for v in saved_videos]

                return (
                    gr.update(visible=False),       # login_section
                    gr.update(visible=True),        # main_section
                    username,                       # logged_in_user
                    f"Welcome, **{username}**",     # welcome_msg
                    "",                             # login_status
                    f"*{user_manager.get_user_count()} registered users*",
                    gr.update(choices=video_choices, value=None),
                    None,                           # video_preview
                    "Upload a video to start",      # upload_status
                    gr.update(interactive=False),   # process_btn
                    helpers.create_placeholder_image("Upload and process "
                                                     "video"),
                    "Upload video, then click Process",
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    None, "",
                    f"Error: {message}",
                    f"*{user_manager.get_user_count()} registered users*",
                    gr.update(choices=[], value=None),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(),
                )

        def do_register(username, password, password_confirm):
            if password != password_confirm:
                return ("Error: Passwords do not match",
                        f"*{user_manager.get_user_count()} registered users*")
            success, message = user_manager.register(username, password)
            return (message,
                    f"*{user_manager.get_user_count()} registered users*")

        def do_logout(username):
            if username and username in ui._sessions:
                del ui._sessions[username]
                logger.info("Removed session for user: %s", username)
            return (
                gr.update(visible=True),        # login_section
                gr.update(visible=False),       # main_section
                None,                           # logged_in_user
                "",                             # welcome_msg
                "",                             # login_username
                "",                             # login_password
                gr.update(choices=[], value=None),
                None,                           # video_preview
                "Upload a video to start",      # upload_status
                gr.update(interactive=False),   # process_btn
            )

        login_btn.click(
            do_login,
            inputs=[login_username, login_password],
            outputs=[login_section, main_section, logged_in_user,
                     welcome_msg, login_status, user_count_info,
                     saved_videos_dropdown, video_preview, upload_status,
                     process_btn, frame_preview, process_status],
        )
        login_password.submit(
            do_login,
            inputs=[login_username, login_password],
            outputs=[login_section, main_section, logged_in_user,
                     welcome_msg, login_status, user_count_info,
                     saved_videos_dropdown, video_preview, upload_status,
                     process_btn, frame_preview, process_status],
        )
        register_btn.click(
            do_register,
            inputs=[reg_username, reg_password, reg_password_confirm],
            outputs=[register_status, user_count_info],
        )
        reg_password_confirm.submit(
            do_register,
            inputs=[reg_username, reg_password, reg_password_confirm],
            outputs=[register_status, user_count_info],
        )
        logout_btn.click(
            do_logout,
            inputs=[logged_in_user],
            outputs=[login_section, main_section, logged_in_user,
                     welcome_msg, login_username, login_password,
                     saved_videos_dropdown, video_preview, upload_status,
                     process_btn],
        )

    return app
