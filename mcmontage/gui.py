"""Desktop app: prepare a replay, finish the render, upload it privately.

    1. Prepare replay  -- pick a .mcpr, write a camera path into it.
                          Minecraft must be CLOSED: it holds replays open.
    2. Finish video    -- pick the rendered .mp4, add music and a title.
    3. Upload          -- private-only, behind a preview-and-confirm gate.

The render itself sits between 1 and 2 and happens inside Minecraft; it cannot
be automated (see README).

Uploads are PRIVATE and privacy is not a control on the form. An unverified API
project has its uploads permanently locked to private with no appeal, so a
"public" option would be a lie -- publishing happens in YouTube Studio.

Threading note: ffmpeg and the packet-stream parse both take a while, so they
run on a worker thread. Tk is not thread-safe, so the worker only ever pushes
strings onto a Queue -- every widget touch happens on the main thread, which
polls that queue with `after`. Calling into Tk from the worker would crash or
hang intermittently.
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import config as config_mod
from . import music as music_mod
from . import pipeline as pipeline_mod
from . import replay as replay_mod

POLL_MS = 100


class App:
    def __init__(self, root):
        self.root = root
        self.cfg = config_mod.load("config.toml")
        self.messages = queue.Queue()
        self.busy = False

        root.title("Minecraft Montage")
        root.geometry("880x680")
        root.minsize(760, 560)

        self.nb = nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        self._build_prepare(nb)
        self._build_finish(nb)
        self._build_upload(nb)

        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_widget = tk.Text(log_frame, height=12, wrap="word",
                                  state="disabled", font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_frame, command=self.log_widget.yview)
        self.log_widget.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_widget.pack(side="left", fill="both", expand=True)

        self.status = ttk.Label(root, text="Ready", anchor="w")
        self.status.pack(fill="x", padx=12, pady=(0, 8))

        self.refresh_replays()
        self.refresh_videos()
        self.refresh_music()
        self.refresh_finished()
        self.root.after(POLL_MS, self._drain)

    # ---------------------------------------------------------------- panels

    def _build_prepare(self, nb):
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="1. Prepare replay")

        ttk.Label(f, text="Do this with Minecraft CLOSED. It writes the camera "
                          "path into the replay; then open the replay in game "
                          "and hit Render Queue → Render.",
                  wraplength=800, foreground="#444").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        ttk.Label(f, text="Replay:").grid(row=1, column=0, sticky="w")
        self.replay_var = tk.StringVar()
        self.replay_combo = ttk.Combobox(f, textvariable=self.replay_var,
                                         width=62, state="readonly")
        self.replay_combo.grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(f, text="Refresh", command=self.refresh_replays).grid(row=1, column=2)
        ttk.Button(f, text="Browse…", command=self.browse_replay).grid(row=1, column=3)

        opts = ttk.LabelFrame(f, text="Camera", padding=10)
        opts.grid(row=2, column=0, columnspan=4, sticky="we", pady=12)

        self.seconds_var = tk.StringVar(value="40")
        self.follow_var = tk.BooleanVar(value=True)
        self.radius_var = tk.StringVar(value="12")
        self.height_var = tk.StringVar(value="6")
        self.revs_var = tk.StringVar(value="0.5")
        self.keyframes_var = tk.StringVar(value="90")

        ttk.Checkbutton(opts, text="Follow the player (recommended)",
                        variable=self.follow_var,
                        command=self._toggle_follow).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        for col, (label, var, tip) in enumerate([
            ("Video length (s)", self.seconds_var, "how long the finished shot is"),
            ("Orbit radius", self.radius_var, "blocks from you"),
            ("Camera height", self.height_var, "blocks above you"),
            ("Revolutions", self.revs_var, "turns over the whole shot"),
        ]):
            ttk.Label(opts, text=label).grid(row=1, column=col, sticky="w", padx=(0, 8))
            ttk.Entry(opts, textvariable=var, width=12).grid(
                row=2, column=col, sticky="w", padx=(0, 8))

        ttk.Label(opts, text="Keyframes").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opts, textvariable=self.keyframes_var, width=12).grid(
            row=4, column=0, sticky="w")

        self.centre_frame = ttk.Frame(opts)
        self.centre_frame.grid(row=3, column=1, columnspan=3, rowspan=2, sticky="w",
                               pady=(8, 0))
        self.cx_var, self.cy_var, self.cz_var = (tk.StringVar() for _ in range(3))
        ttk.Label(self.centre_frame,
                  text="Static orbit centre (only when not following):").grid(
            row=0, column=0, columnspan=6, sticky="w")
        for i, (lbl, var) in enumerate([("X", self.cx_var), ("Y", self.cy_var),
                                        ("Z", self.cz_var)]):
            ttk.Label(self.centre_frame, text=lbl).grid(row=1, column=i * 2, padx=(0, 3))
            ttk.Entry(self.centre_frame, textvariable=var, width=10).grid(
                row=1, column=i * 2 + 1, padx=(0, 8))

        self.prepare_btn = ttk.Button(f, text="Write camera path into replay",
                                      command=self.do_prepare)
        self.prepare_btn.grid(row=3, column=0, columnspan=2, sticky="w")

        f.columnconfigure(1, weight=1)
        self._toggle_follow()

    def _build_finish(self, nb):
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="2. Finish video")

        ttk.Label(f, text="After rendering in Minecraft, pick the rendered file "
                          "and add music and a title. Safe to run with "
                          "Minecraft open or closed.",
                  wraplength=800, foreground="#444").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        ttk.Label(f, text="Video:").grid(row=1, column=0, sticky="w")
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(f, textvariable=self.video_var,
                                        width=62, state="readonly")
        self.video_combo.grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(f, text="Refresh", command=self.refresh_videos).grid(row=1, column=2)
        ttk.Button(f, text="Browse…", command=self.browse_video).grid(row=1, column=3)

        ttk.Label(f, text="Title:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.title_var = tk.StringVar(value="Harrison's Long-Haul World")
        ttk.Entry(f, textvariable=self.title_var).grid(
            row=2, column=1, columnspan=3, sticky="we", padx=6, pady=(10, 0))

        ttk.Label(f, text="Subtitle:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.subtitle_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.subtitle_var).grid(
            row=3, column=1, columnspan=3, sticky="we", padx=6, pady=(6, 0))

        ttk.Label(f, text="Music:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.music_var = tk.StringVar()
        self.music_combo = ttk.Combobox(f, textvariable=self.music_var,
                                        width=40, state="readonly")
        self.music_combo.grid(row=4, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Button(f, text="Refresh", command=self.refresh_music).grid(
            row=4, column=2, pady=(6, 0))

        self.finish_btn = ttk.Button(f, text="Build finished video",
                                     command=self.do_finish)
        self.finish_btn.grid(row=5, column=0, columnspan=2, sticky="w", pady=(16, 0))
        ttk.Button(f, text="Open output folder",
                   command=self.open_output).grid(row=5, column=2, pady=(16, 0))

        f.columnconfigure(1, weight=1)

    def _build_upload(self, nb):
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="3. Upload (private)")

        ttk.Label(
            f,
            text="Uploads land PRIVATE and stay private. An unverified API "
                 "project cannot publish publicly, so publishing is always a "
                 "separate step you take in YouTube Studio. Watch the video "
                 "first — the Preview button opens it.",
            wraplength=800, foreground="#444").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        ttk.Label(f, text="Video:").grid(row=1, column=0, sticky="w")
        self.upload_var = tk.StringVar()
        self.upload_combo = ttk.Combobox(f, textvariable=self.upload_var,
                                         width=58, state="readonly")
        self.upload_combo.grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(f, text="Refresh", command=self.refresh_finished).grid(row=1, column=2)
        ttk.Button(f, text="Browse…", command=self.browse_finished).grid(row=1, column=3)

        ttk.Label(f, text="Title:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.up_title_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.up_title_var).grid(
            row=2, column=1, columnspan=3, sticky="we", padx=6, pady=(10, 0))

        ttk.Label(f, text="Description:").grid(row=3, column=0, sticky="nw", pady=(6, 0))
        self.up_desc = tk.Text(f, height=6, wrap="word")
        self.up_desc.grid(row=3, column=1, columnspan=3, sticky="we", padx=6, pady=(6, 0))

        ttk.Label(f, text="Tags:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.up_tags_var = tk.StringVar(value="minecraft, timelapse, neoforge")
        ttk.Entry(f, textvariable=self.up_tags_var).grid(
            row=4, column=1, columnspan=3, sticky="we", padx=6, pady=(6, 0))

        ttk.Label(f, text="Music used:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.up_music_var = tk.StringVar()
        self.up_music_combo = ttk.Combobox(f, textvariable=self.up_music_var,
                                           width=40, state="readonly")
        self.up_music_combo.grid(row=5, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(f, text="(its credit is added to the description "
                          "automatically)", foreground="#666").grid(
            row=5, column=2, columnspan=2, sticky="w", pady=(6, 0))

        btns = ttk.Frame(f)
        btns.grid(row=6, column=0, columnspan=4, sticky="we", pady=(16, 0))
        ttk.Button(btns, text="▶ Preview video",
                   command=self.preview_upload).pack(side="left")
        ttk.Button(btns, text="Show final description",
                   command=self.show_description).pack(side="left", padx=8)
        self.upload_btn = ttk.Button(btns, text="Upload privately to YouTube",
                                     command=self.do_upload)
        self.upload_btn.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(f, mode="determinate", maximum=100)
        self.progress.grid(row=7, column=0, columnspan=4, sticky="we", pady=(12, 0))

        self.studio_url = None
        self.studio_btn = ttk.Button(f, text="Open in YouTube Studio to publish",
                                     command=self.open_studio, state="disabled")
        self.studio_btn.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))

        f.columnconfigure(1, weight=1)

    # -------------------------------------------------------------- helpers

    def _toggle_follow(self):
        state = "disabled" if self.follow_var.get() else "normal"
        for child in self.centre_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def log(self, msg=""):
        self.messages.put(("log", str(msg)))

    def _drain(self):
        """Main-thread pump. The worker never touches Tk directly."""
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self.log_widget.configure(state="normal")
                    self.log_widget.insert("end", payload + "\n")
                    self.log_widget.see("end")
                    self.log_widget.configure(state="disabled")
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "finished":
                    # pre-fill the upload tab with what was just built
                    path_str, title, music_id = payload
                    self.refresh_finished()
                    for i, p in enumerate(self._finished):
                        if str(p.resolve()) == path_str:
                            self.upload_combo.current(i)
                            break
                    if title and not self.up_title_var.get():
                        self.up_title_var.set(title)
                    if music_id:
                        for i, t in enumerate(self._music):
                            if t.id == music_id:
                                self.up_music_combo.current(i + 1)
                                break
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "studio":
                    self.studio_url = payload
                    self.studio_btn.configure(state="normal")
                elif kind == "done":
                    self.busy = False
                    self.prepare_btn.configure(state="normal")
                    self.finish_btn.configure(state="normal")
                    self.upload_btn.configure(state="normal")
                elif kind == "error":
                    messagebox.showerror("Something went wrong", payload)
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._drain)

    def _run(self, fn, status):
        if self.busy:
            messagebox.showinfo("Busy", "Something is already running.")
            return
        self.busy = True
        self.prepare_btn.configure(state="disabled")
        self.finish_btn.configure(state="disabled")
        self.upload_btn.configure(state="disabled")
        self.messages.put(("status", status))

        def worker():
            try:
                fn()
                self.messages.put(("status", "Ready"))
            except Exception as exc:
                self.messages.put(("log", ""))
                self.messages.put(("log", "ERROR: %s" % exc))
                self.messages.put(("log", traceback.format_exc()))
                self.messages.put(("status", "Failed"))
                self.messages.put(("error", str(exc)))
            finally:
                self.messages.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------- populate

    def refresh_replays(self):
        self._replays = replay_mod.find_replays(self.cfg.recordings)
        labels = ["%s  (%.1f MB)" % (p.name, p.stat().st_size / 1e6)
                  for p in self._replays]
        self.replay_combo["values"] = labels
        if labels and not self.replay_var.get():
            self.replay_combo.current(0)
        self.log("Found %d replay file(s) in %s" % (len(labels), self.cfg.recordings))

    def refresh_videos(self):
        folder = Path(self.cfg.videos)
        self._videos = (
            sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if folder.is_dir() else []
        )
        labels = ["%s  (%.1f MB)" % (p.name, p.stat().st_size / 1e6)
                  for p in self._videos]
        self.video_combo["values"] = labels
        if labels and not self.video_var.get():
            self.video_combo.current(0)
        self.log("Found %d rendered video(s) in %s" % (len(labels), folder))

    def refresh_music(self):
        try:
            self._music = music_mod.load_library()
        except music_mod.MusicError as exc:
            self._music = []
            self.log("Music library: %s" % exc)
        labels = ["(no music)"] + ["%s — %s" % (t.id, t.title) for t in self._music]
        self.music_combo["values"] = labels
        if not self.music_var.get():
            self.music_combo.current(1 if self._music else 0)

    def _selected(self, combo, items):
        idx = combo.current()
        return items[idx] if 0 <= idx < len(items) else None

    def browse_replay(self):
        p = filedialog.askopenfilename(
            title="Choose a replay", initialdir=str(self.cfg.recordings),
            filetypes=[("Minecraft replay", "*.mcpr"), ("All files", "*.*")])
        if p:
            self._replays = [Path(p)] + list(self._replays)
            self.replay_combo["values"] = [Path(p).name] + list(self.replay_combo["values"])
            self.replay_combo.current(0)

    def browse_video(self):
        p = filedialog.askopenfilename(
            title="Choose a rendered video", initialdir=str(self.cfg.videos),
            filetypes=[("Video", "*.mp4 *.mkv *.webm"), ("All files", "*.*")])
        if p:
            self._videos = [Path(p)] + list(self._videos)
            self.video_combo["values"] = [Path(p).name] + list(self.video_combo["values"])
            self.video_combo.current(0)

    def open_output(self):
        out = Path(self.cfg.output).resolve()
        out.mkdir(parents=True, exist_ok=True)
        import os
        os.startfile(str(out))  # noqa: S606 - Windows shell open

    def refresh_finished(self):
        folder = Path(self.cfg.output)
        self._finished = (
            sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if folder.is_dir() else []
        )
        labels = ["%s  (%.1f MB)" % (p.name, p.stat().st_size / 1e6)
                  for p in self._finished]
        self.upload_combo["values"] = labels
        if labels and not self.upload_var.get():
            self.upload_combo.current(0)
        self.up_music_combo["values"] = (
            ["(none)"] + ["%s — %s" % (t.id, t.title) for t in self._music]
        )
        if not self.up_music_var.get():
            self.up_music_combo.current(0)

    def browse_finished(self):
        p = filedialog.askopenfilename(
            title="Choose a finished video",
            initialdir=str(Path(self.cfg.output).resolve()),
            filetypes=[("Video", "*.mp4 *.mkv *.webm"), ("All files", "*.*")])
        if p:
            self._finished = [Path(p)] + list(self._finished)
            self.upload_combo["values"] = (
                [Path(p).name] + list(self.upload_combo["values"]))
            self.upload_combo.current(0)

    def _upload_selection(self):
        video = self._selected(self.upload_combo, self._finished)
        idx = self.up_music_combo.current()
        music_id = self._music[idx - 1].id if idx > 0 and self._music else None
        return video, music_id

    def preview_upload(self):
        video, _ = self._upload_selection()
        if video is None:
            messagebox.showwarning("No video", "Pick a video first.")
            return
        import os
        os.startfile(str(video))  # noqa: S606 - Windows shell open

    def show_description(self):
        _, music_id = self._upload_selection()
        body = self.up_desc.get("1.0", "end").strip()
        try:
            text = pipeline_mod.build_upload_description(body, music_id)
        except music_mod.MusicError as exc:
            messagebox.showerror("Music problem", str(exc))
            return
        self.log("")
        self.log("--- description as it will be uploaded ---")
        for line in text.splitlines() or [""]:
            self.log("  " + line)
        self.log("--- end ---")

    def open_studio(self):
        if self.studio_url:
            import webbrowser
            webbrowser.open(self.studio_url)

    # -------------------------------------------------------------- actions

    def do_prepare(self):
        replay = self._selected(self.replay_combo, self._replays)
        if replay is None:
            messagebox.showwarning("No replay", "Pick a replay first.")
            return
        if pipeline_mod.minecraft_running():
            messagebox.showwarning(
                "Minecraft is running",
                "Minecraft holds replay files open, so writing the camera path "
                "would fail.\n\nQuit Minecraft to the desktop, then try again.")
            return

        try:
            seconds = float(self.seconds_var.get())
            radius = float(self.radius_var.get())
            height = float(self.height_var.get())
            revs = float(self.revs_var.get())
            keyframes = int(self.keyframes_var.get())
            follow = self.follow_var.get()
            centre = None
            if not follow:
                centre = (float(self.cx_var.get()), float(self.cy_var.get()),
                          float(self.cz_var.get()))
        except ValueError as exc:
            messagebox.showerror("Check the numbers", str(exc))
            return

        def job():
            self.log("")
            self.log("=== Preparing %s ===" % replay.name)
            summary = pipeline_mod.plan_replay(
                self.cfg, replay, seconds=seconds, follow=follow, centre=centre,
                radius=radius, height=height, revolutions=revs,
                keyframes=keyframes, log=self.log,
            )
            self.log("Speed-up x%.1f over %.1f min of replay."
                     % (summary["speedup"], (summary["end"] - summary["start"]) / 60000))
            self.log("Renders to: %s" % summary["out_file"])
            self.log("")
            self.log("Next: start Minecraft → Replay Viewer → open this "
                     "replay → Render Queue → Render.")

        self._run(job, "Preparing replay…")

    def do_upload(self):
        video, music_id = self._upload_selection()
        if video is None:
            messagebox.showwarning("No video", "Pick a video first.")
            return
        title = self.up_title_var.get().strip()
        if not title:
            messagebox.showwarning("No title", "Give the video a title.")
            return
        body = self.up_desc.get("1.0", "end").strip()
        tags = [t for t in self.up_tags_var.get().split(",")]

        try:
            description = pipeline_mod.build_upload_description(body, music_id)
        except music_mod.MusicError as exc:
            messagebox.showerror("Music problem", str(exc))
            return

        # The review gate: say plainly what is about to leave the machine.
        preview = description if len(description) < 500 else description[:500] + "…"
        if not messagebox.askokcancel(
            "Upload to YouTube?",
            "This uploads to your YouTube account as PRIVATE.\n\n"
            "File:  %s (%.1f MB)\n"
            "Title: %s\n\n"
            "Description:\n%s\n\n"
            "It will NOT be public. Publishing stays a separate step you take "
            "in YouTube Studio afterwards."
            % (video.name, video.stat().st_size / 1e6, title, preview),
        ):
            self.log("Upload cancelled.")
            return

        self.studio_btn.configure(state="disabled")
        self.progress["value"] = 0

        def job():
            self.log("")
            self.log("=== Uploading %s ===" % video.name)

            def on_progress(fraction):
                self.messages.put(("progress", fraction * 100))

            resp = pipeline_mod.upload_private(
                self.cfg, video, title=title, description=body,
                tags=tags, music_id=music_id,
                progress=on_progress, log=self.log,
            )
            from . import youtube as yt_mod
            self.messages.put(("studio", yt_mod.studio_url(resp)))

        self._run(job, "Uploading to YouTube…")

    def do_finish(self):
        video = self._selected(self.video_combo, self._videos)
        if video is None:
            messagebox.showwarning("No video", "Pick a rendered video first.")
            return
        idx = self.music_combo.current()
        music_id = self._music[idx - 1].id if idx > 0 and self._music else None
        title = self.title_var.get().strip()
        subtitle = self.subtitle_var.get().strip()

        def job():
            self.log("")
            self.log("=== Finishing %s ===" % video.name)
            out, _track = pipeline_mod.finish_video(
                self.cfg, [video], title=title, subtitle=subtitle,
                music_id=music_id, log=self.log,
            )
            self.log("")
            self.log("Saved to %s" % out.resolve())
            self.messages.put(("finished", (str(out.resolve()), title, music_id)))

        self._run(job, "Building video…")


# Held for the process lifetime; releasing it is what frees the single-instance
# lock, so this must stay referenced at module scope.
_instance_lock = None


def _claim_single_instance(port=49731):
    """True if this is the only instance.

    Binding a fixed loopback port is the simplest lock that cleans itself up:
    a lock *file* survives a crash and then blocks every future launch, whereas
    the OS drops this the moment the process dies. Nothing ever connects to it.
    """
    global _instance_lock
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
    except OSError:
        s.close()
        return False
    _instance_lock = s
    return True


def main():
    if not _claim_single_instance():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Already running",
            "Minecraft Montage is already open.\n\n"
            "Look for its window (check the taskbar).")
        root.destroy()
        return

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    try:
        root.iconbitmap(str(Path(__file__).resolve().parent.parent / "icon.ico"))
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
