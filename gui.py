import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import os
import time
import settings

_THUMB = (480, 360)  # default thumbnail size for each video panel when no size is available yet
_NAMES = ["Camera", "LSD", "Warped", "Filtered"]  # display names for the four video panels
_POSITIONS = {"Camera": (0, 0), "LSD": (0, 1), "Warped": (1, 0), "Filtered": (1, 1)}  # grid position (row, col) of each panel

_root = None
_video_area = None
_ctrl_frame = None
_frames = {}
_containers = {}
_labels = {}
_photos = {}
_last_imgs = {}
_closed = False
_focused = None
_status_var = None
_fullscreen = False

_var_paused = None
_var_filter = None
_var_clahe_clip = None
_var_clahe_tile = None
_var_sharpen = None
_var_show_intersections = None
_var_camera = None
_var_lsd_v_min_deg = None
_var_lsd_candidates = None
_var_lsd_match_tol = None

_recorder = None
_recording = False
_rec_btn = None
_rec_frame_size = None


def _on_close():
    global _closed, _recorder
    _closed = True
    if _recorder is not None:
        _recorder.release()
        _recorder = None
    # Do not save settings automatically on close
    _root.quit()


def _panel_size():
    window_width = max(_video_area.winfo_width(), _THUMB[0] * 2 + 24)  # at least two thumbs wide with padding
    window_height = max(_video_area.winfo_height(), _THUMB[1] * 2 + 24)  # at least two thumbs tall with padding
    if _focused:
        return window_width - 12, window_height - 36  # one panel fills nearly the whole video area
    return window_width // 2 - 12, window_height // 2 - 36  # four panels share the video area equally


def _render(name, img, size):
    if img is None:
        _labels[name].config(image="")  # clear the panel if no image is available
        return
    if len(img.shape) == 2:
        image_for_tk = Image.fromarray(img)  # grayscale — PIL accepts it directly
    else:
        image_for_tk = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert BGR (OpenCV) to RGB (PIL)
    image_for_tk = image_for_tk.resize(size, Image.BILINEAR)  # scale to the panel size
    tk_photo = ImageTk.PhotoImage(image_for_tk)
    _photos[name] = tk_photo  # keep a reference — Tkinter does not hold one itself
    _labels[name].config(image=tk_photo)
    _containers[name].config(width=size[0], height=size[1])


def _redraw_all():
    size = _panel_size()
    for name in _NAMES:
        _render(name, _last_imgs.get(name), size)


def _toggle_focus(name):
    global _focused
    if _focused == name:  # clicking the focused panel again returns to the 4-panel view
        _focused = None
        for panel_name in _NAMES:
            row, column = _POSITIONS[panel_name]
            _frames[panel_name].grid(row=row, column=column, padx=4, pady=4, sticky="nsew")  # restore all panels to their grid positions
    else:
        _focused = name  # expand this panel to fill the video area
        for panel_name in _NAMES:
            if panel_name != name:
                _frames[panel_name].grid_remove()  # hide the other panels without destroying them
            else:
                _frames[panel_name].grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
    _redraw_all()


def _on_resize(event):
    if event.widget is _video_area:
        _redraw_all()


def _toggle_fullscreen(event=None):
    global _fullscreen
    _fullscreen = not _fullscreen
    _root.attributes("-fullscreen", _fullscreen)


def _sync_setting_from_var(key, var, cast=None):
    try:
        value = var.get()
        if cast:
            value = cast(value)
        settings.set_val(key, value)
    except (tk.TclError, ValueError):
        pass


def _snapshot():
    for panel_name in ["Filtered", "Warped", "Camera"]:  # prefer the best-processed image available
        image = _last_imgs.get(panel_name)
        if image is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")  # timestamp for unique filename
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=f"camscreen_{panel_name}_{timestamp}.png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
            )
            if file_path:
                cv2.imwrite(file_path, image)  # save as-is (BGR for colour, uint8 for grayscale)
                _status_var.set(f"Saved {os.path.basename(file_path)}")
            return  # only save the first available image


def _toggle_record():
    global _recorder, _recording
    if _recording:
        if _recorder is not None:
            _recorder.release()
            _recorder = None
        global _rec_frame_size
        _rec_frame_size = None
        _recording = False
        _rec_btn.config(text="Record")
        _status_var.set("Recording stopped")
    else:
        for panel_name in ["Filtered", "Warped", "Camera"]:
            image = _last_imgs.get(panel_name)
            if image is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".mp4",
                    initialfile=f"camscreen_{timestamp}.mp4",
                    filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")],
                )
                if file_path:
                    image_height, image_width = image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # Always open writer in BGR colour mode and fixed frame size
                    _rec_frame_size = (image_width, image_height)
                    _recorder = cv2.VideoWriter(file_path, fourcc, 20.0, _rec_frame_size, True)
                    _recording = True
                    _rec_btn.config(text="Stop Rec")
                    _status_var.set(f"Recording to {os.path.basename(file_path)}")
                return


def _write_record_frame(img):
    global _recorder, _rec_frame_size
    if _recorder is None or _rec_frame_size is None or img is None:
        return
    image_height, image_width = img.shape[:2]
    saved_width, saved_height = _rec_frame_size  # fixed size set when recording started
    if len(img.shape) == 2:
        frame_for_video = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # VideoWriter requires 3-channel BGR
    else:
        frame_for_video = img
    if (image_width, image_height) != (saved_width, saved_height):
        frame_for_video = cv2.resize(frame_for_video, (saved_width, saved_height))  # resize to match the writer's fixed frame size
    try:
        _recorder.write(frame_for_video)  # append the frame to the video file
    except Exception:
        pass  # silently drop frames on write error


def _reset_detection():
    settings.set_val("_reset_flag", True)
    _status_var.set("Detection reset")


def _build_controls(parent):
    global _var_paused, _var_filter
    global _var_clahe_clip, _var_clahe_tile, _var_sharpen
    global _var_show_intersections
    global _var_camera, _rec_btn
    global _var_lsd_v_min_deg, _var_lsd_candidates, _var_lsd_match_tol

    style = {"bg": "#2d2d2d", "fg": "#cccccc", "font": ("Helvetica", 9)}
    slider_style = {"bg": "#2d2d2d", "fg": "#cccccc", "troughcolor": "#444",
                    "highlightthickness": 0, "font": ("Helvetica", 8)}
    section_style = {"bg": "#2d2d2d", "fg": "#88aaff", "font": ("Helvetica", 10, "bold")}
    btn_style = {"bg": "#444", "fg": "#eee", "font": ("Helvetica", 9), "relief": "flat",
                 "activebackground": "#666", "activeforeground": "#fff", "cursor": "hand2"}

    parent.configure(bg="#2d2d2d")
    canvas = tk.Canvas(parent, bg="#2d2d2d", highlightthickness=0, width=220)
    scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg="#2d2d2d")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _section(text):
        tk.Label(inner, text=text, **section_style).pack(anchor="w", padx=6, pady=(10, 2))
        ttk.Separator(inner, orient="horizontal").pack(fill="x", padx=6)

    def _slider(label, from_, to, var, res=1):
        row_frame = tk.Frame(inner, bg="#2d2d2d")
        row_frame.pack(fill="x", padx=6, pady=2)
        tk.Label(row_frame, text=label, **style).pack(side=tk.LEFT)
        tk.Scale(row_frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var,
                 resolution=res, length=130, **slider_style).pack(side=tk.RIGHT)

    def _check(label, var):
        tk.Checkbutton(inner, text=label, variable=var, onvalue=True, offvalue=False,
                       bg="#2d2d2d", fg="#cccccc", selectcolor="#444",
                       activebackground="#2d2d2d", activeforeground="#cccccc",
                       font=("Helvetica", 9)).pack(anchor="w", padx=6, pady=1)

    # ── Camera ──
    _section("Camera")
    _var_camera = tk.IntVar(value=settings.get("camera_index"))
    row_frame = tk.Frame(inner, bg="#2d2d2d")
    row_frame.pack(fill="x", padx=6, pady=2)
    tk.Label(row_frame, text="Camera #", **style).pack(side=tk.LEFT)
    tk.Spinbox(row_frame, from_=0, to=9, textvariable=_var_camera, width=4,
               bg="#444", fg="#eee", font=("Helvetica", 9),
               command=lambda: _sync_setting_from_var("camera_index", _var_camera, int)).pack(side=tk.RIGHT)

    _var_paused = tk.BooleanVar(value=settings.get("paused"))
    tk.Button(inner, text="Pause / Resume",
              command=lambda: (_var_paused.set(not _var_paused.get()),
                               settings.set_val("paused", _var_paused.get()),
                               _status_var.set("Paused" if _var_paused.get() else "Running")),
              **btn_style).pack(fill="x", padx=6, pady=4)

    # ── LSD Detection ──
    _section("LSD Detection")
    _var_lsd_v_min_deg   = tk.IntVar(value=settings.get("lsd_v_min_deg"))
    _var_lsd_candidates  = tk.IntVar(value=settings.get("lsd_v_candidates"))
    _var_lsd_match_tol   = tk.DoubleVar(value=settings.get("lsd_v_match_tol"))
    _slider("V-Min Angle",  30,  89, _var_lsd_v_min_deg)
    _slider("Candidates",    2,  20, _var_lsd_candidates)
    _slider("Match Tol",  0.05, 1.0, _var_lsd_match_tol, 0.05)
    _var_lsd_v_min_deg.trace_add("write",  lambda *_: _sync_setting_from_var("lsd_v_min_deg",   _var_lsd_v_min_deg,  int))
    _var_lsd_candidates.trace_add("write", lambda *_: _sync_setting_from_var("lsd_v_candidates", _var_lsd_candidates, int))
    _var_lsd_match_tol.trace_add("write",  lambda *_: _sync_setting_from_var("lsd_v_match_tol",  _var_lsd_match_tol,  float))

    # ── Filter ──
    _section("Image Filter")
    _var_filter = tk.BooleanVar(value=settings.get("filter_enabled"))
    _check("Enable Filter", _var_filter)
    _var_filter.trace_add("write", lambda *_: _sync_setting_from_var("filter_enabled", _var_filter, bool))

    _var_clahe_clip = tk.DoubleVar(value=settings.get("clahe_clip"))
    _var_clahe_tile = tk.IntVar(value=settings.get("clahe_tile"))
    _var_sharpen = tk.IntVar(value=settings.get("sharpen_strength"))
    _slider("CLAHE Clip", 0.5, 10.0, _var_clahe_clip, 0.5)
    _slider("CLAHE Tile", 2, 32, _var_clahe_tile, 2)
    _slider("Sharpen", 1, 15, _var_sharpen)
    _var_clahe_clip.trace_add("write", lambda *_: _sync_setting_from_var("clahe_clip", _var_clahe_clip, float))
    _var_clahe_tile.trace_add("write", lambda *_: _sync_setting_from_var("clahe_tile", _var_clahe_tile, int))
    _var_sharpen.trace_add("write", lambda *_: _sync_setting_from_var("sharpen_strength", _var_sharpen, int))

    # ── Debug Overlays ──
    _section("Debug Overlays")
    _var_show_intersections = tk.BooleanVar(value=settings.get("show_intersections"))
    _check("Show Intersections", _var_show_intersections)
    _var_show_intersections.trace_add("write", lambda *_: _sync_setting_from_var("show_intersections", _var_show_intersections, bool))

    # ── Actions ──
    _section("Actions")
    tk.Button(inner, text="Snapshot", command=_snapshot, **btn_style).pack(fill="x", padx=6, pady=2)
    _rec_btn = tk.Button(inner, text="Record", command=_toggle_record, **btn_style)
    _rec_btn.pack(fill="x", padx=6, pady=2)
    tk.Button(inner, text="Reset Detection", command=_reset_detection, **btn_style).pack(fill="x", padx=6, pady=2)
    tk.Button(inner, text="Fullscreen (F11)", command=_toggle_fullscreen, **btn_style).pack(fill="x", padx=6, pady=2)
    tk.Button(inner, text="Revert To Saved", command=lambda: (settings.restore_saved(), _load_vars_from_settings(), _status_var.set("Reverted to last saved settings")), **btn_style).pack(fill="x", padx=6, pady=2)
    tk.Button(inner, text="Save Settings", command=lambda: (settings.save(), _status_var.set("Settings saved")), **btn_style).pack(fill="x", padx=6, pady=2)
    tk.Button(inner, text="Reset To Defaults", command=lambda: (settings.reset(), _load_vars_from_settings(), _status_var.set("Settings reset to defaults")), **btn_style).pack(fill="x", padx=6, pady=2)
    # Load current settings into the UI vars so controls reflect stored values
    try:
        _load_vars_from_settings()
    except Exception:
        pass


def _init():
    global _root, _video_area, _ctrl_frame, _status_var

    _root = tk.Tk()
    _root.title("CamScreen")
    _root.configure(bg="#1e1e1e")
    _root.geometry(f"{_THUMB[0]*2+260}x{_THUMB[1]*2+100}")
    _root.minsize(800, 500)
    _root.resizable(True, True)
    _root.protocol("WM_DELETE_WINDOW", _on_close)

    _root.bind("<Escape>", lambda e: _on_close())
    _root.bind("<F11>", _toggle_fullscreen)
    _root.bind("<space>", lambda e: (_var_paused.set(not _var_paused.get()),
                                     settings.set_val("paused", _var_paused.get())))
    _root.bind("s", lambda e: _snapshot())
    _root.bind("f", lambda e: (settings.set_val("filter_enabled", not settings.get("filter_enabled")),
                                _var_filter.set(settings.get("filter_enabled"))))

    main_window = tk.PanedWindow(_root, orient=tk.HORIZONTAL, bg="#1e1e1e", sashwidth=4)
    main_window.pack(fill=tk.BOTH, expand=True)

    _video_area = tk.Frame(main_window, bg="#1e1e1e")
    main_window.add(_video_area, stretch="always")

    _video_area.columnconfigure(0, weight=1)
    _video_area.columnconfigure(1, weight=1)
    _video_area.rowconfigure(0, weight=1)
    _video_area.rowconfigure(1, weight=1)

    for name, (row, col) in _POSITIONS.items():
        outer = tk.Frame(_video_area, bg="#1e1e1e")
        outer.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        tk.Label(outer, text=name, bg="#1e1e1e", fg="#aaaaaa",
                 font=("Helvetica", 10, "bold")).pack(side=tk.TOP, anchor="w")
        container = tk.Frame(outer, bg="#2d2d2d", width=_THUMB[0], height=_THUMB[1])
        container.pack(fill=tk.BOTH, expand=True)
        container.pack_propagate(False)
        image_label = tk.Label(container, bg="#2d2d2d", cursor="hand2")
        image_label.pack(fill=tk.BOTH, expand=True)
        image_label.bind("<Button-1>", lambda e, n=name: _toggle_focus(n))
        _frames[name] = outer
        _containers[name] = container
        _labels[name] = image_label
        _last_imgs[name] = None

    _ctrl_frame = tk.Frame(main_window, bg="#2d2d2d", width=240)
    main_window.add(_ctrl_frame, stretch="never")

    _build_controls(_ctrl_frame)

    _status_var = tk.StringVar(value="Ready")
    status_bar = tk.Label(_root, textvariable=_status_var, bg="#181818", fg="#77cc77",
                          font=("Helvetica", 9), anchor="w", padx=8, pady=3)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    _video_area.bind("<Configure>", _on_resize)


def _load_vars_from_settings():
    try:
        if _var_camera is not None:
            _var_camera.set(settings.get("camera_index"))
        if _var_paused is not None:
            _var_paused.set(settings.get("paused"))
        if _var_filter is not None:
            _var_filter.set(settings.get("filter_enabled"))
        if _var_clahe_clip is not None:
            _var_clahe_clip.set(settings.get("clahe_clip"))
        if _var_clahe_tile is not None:
            _var_clahe_tile.set(settings.get("clahe_tile"))
        if _var_sharpen is not None:
            _var_sharpen.set(settings.get("sharpen_strength"))
        if _var_show_intersections is not None:
            _var_show_intersections.set(settings.get("show_intersections"))
        if _var_lsd_v_min_deg is not None:
            _var_lsd_v_min_deg.set(settings.get("lsd_v_min_deg"))
        if _var_lsd_candidates is not None:
            _var_lsd_candidates.set(settings.get("lsd_v_candidates"))
        if _var_lsd_match_tol is not None:
            _var_lsd_match_tol.set(settings.get("lsd_v_match_tol"))
    except Exception:
        pass


def show_windows(frame, canny, warped=None, filtered=None, fps=0.0, det_status="") -> bool:
    global _root, _recorder
    if _root is None:
        _init()
    if _closed:
        return False

    _last_imgs["Camera"] = frame
    _last_imgs["LSD"] = canny
    _last_imgs["Warped"] = warped
    _last_imgs["Filtered"] = filtered

    panel_size = _panel_size()
    for panel_name in (_NAMES if not _focused else [_focused]):
        _render(panel_name, _last_imgs[panel_name], panel_size)

    if fps > 0 or det_status:
        status_parts = []
        if fps > 0:
            status_parts.append(f"FPS: {fps:.1f}")
        if det_status:
            status_parts.append(det_status)
        if _recording:
            status_parts.append("REC")
        if _var_paused and _var_paused.get():
            status_parts.append("PAUSED")
        _status_var.set("  |  ".join(status_parts))

    if _recording and _recorder is not None:
        for panel_name in ["Filtered", "Warped", "Camera"]:
            image = _last_imgs.get(panel_name)
            if image is not None:
                _write_record_frame(image)
                break

    _root.update_idletasks()
    _root.update()
    return not _closed
