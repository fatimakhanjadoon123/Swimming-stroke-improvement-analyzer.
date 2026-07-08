"""
Swimming Stroke Improvement Analyzer
-------------------------------------
Project brief: "Analyze form from race videos."
Tech: Tkinter | Category: Video analysis | Difficulty: Advanced

Load a race video, watch it play back with a live pose skeleton overlay,
then generate a stroke-analysis report (stroke rate, elbow/catch angles,
body roll, left/right symmetry) with charts and plain-language feedback.

Run with:  python main.py
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from video_processor import VideoProcessor
from stroke_analyzer import StrokeAnalyzer


class SwimAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Swimming Stroke Improvement Analyzer")
        self.geometry("1180x760")
        self.minsize(1000, 680)
        self.configure(bg="#0d1b2a")

        self.video_path = None
        self.processor = None
        self.analyzer = None
        self.playing = False
        self.play_thread = None
        self.stop_flag = threading.Event()
        self.current_photo = None  # keep reference to avoid GC

        self._build_style()
        self._build_layout()

    # ---------------- UI construction ----------------

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#0d1b2a")
        style.configure("TLabel", background="#0d1b2a", foreground="#e0e1dd", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#48cae4")
        style.configure("Metric.TLabel", font=("Segoe UI", 11), foreground="#e0e1dd")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TProgressbar", troughcolor="#1b263b", background="#48cae4")

    def _build_layout(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(14, 6))
        ttk.Label(header, text="🏊 Swimming Stroke Improvement Analyzer", style="Header.TLabel").pack(side="left")

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=16, pady=6)
        ttk.Button(controls, text="Load Race Video", command=self.load_video).pack(side="left", padx=4)
        self.play_btn = ttk.Button(controls, text="▶ Play", command=self.toggle_play, state="disabled")
        self.play_btn.pack(side="left", padx=4)
        self.analyze_btn = ttk.Button(controls, text="Analyze Full Video", command=self.run_full_analysis, state="disabled")
        self.analyze_btn.pack(side="left", padx=4)
        self.export_btn = ttk.Button(controls, text="Export Report", command=self.export_report, state="disabled")
        self.export_btn.pack(side="left", padx=4)

        self.file_label = ttk.Label(controls, text="No video loaded")
        self.file_label.pack(side="left", padx=16)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=16, pady=6)

        # Left: video canvas
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.video_label = tk.Label(left, bg="#1b263b", text="Load a video to begin",
                                     fg="#8d99ae", font=("Segoe UI", 12))
        self.video_label.pack(fill="both", expand=True)

        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 0))

        # Right: metrics + report panel
        right = ttk.Frame(body, width=380)
        right.pack(side="right", fill="y", padx=(16, 0))
        right.pack_propagate(False)

        ttk.Label(right, text="Analysis Report", style="Header.TLabel").pack(anchor="w", pady=(0, 8))
        self.metrics_text = tk.Text(right, height=14, width=42, bg="#1b263b", fg="#e0e1dd",
                                     font=("Consolas", 10), relief="flat", wrap="word")
        self.metrics_text.pack(fill="x")
        self.metrics_text.insert("end", "Load a video, then click 'Analyze Full Video'\n"
                                         "to generate stroke metrics and feedback.")
        self.metrics_text.config(state="disabled")

        self.chart_frame = ttk.Frame(right)
        self.chart_frame.pack(fill="both", expand=True, pady=(10, 0))

        status = ttk.Frame(self)
        status.pack(fill="x", padx=16, pady=(0, 10))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

    # ---------------- video loading / playback ----------------

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select a swimming race video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            if self.processor:
                self.processor.release()
            self.processor = VideoProcessor(path)
        except IOError as e:
            messagebox.showerror("Error", str(e))
            return

        self.video_path = path
        self.analyzer = StrokeAnalyzer(fps=self.processor.fps)
        self.file_label.config(text=os.path.basename(path))
        self.play_btn.config(state="normal")
        self.analyze_btn.config(state="normal")
        self.export_btn.config(state="disabled")
        self.progress.config(maximum=self.processor.frame_count, value=0)
        self.status_var.set(f"Loaded video: {self.processor.width}x{self.processor.height} "
                             f"@ {self.processor.fps:.1f} fps, {self.processor.frame_count} frames")
        self._show_single_frame()

    def _show_single_frame(self):
        ok, frame, _ = self.processor.read_frame()
        if ok:
            self._display_frame(frame)
        self.processor.seek(0)

    def toggle_play(self):
        if not self.processor:
            return
        if self.playing:
            self.stop_flag.set()
            self.playing = False
            self.play_btn.config(text="▶ Play")
        else:
            self.stop_flag.clear()
            self.playing = True
            self.play_btn.config(text="⏸ Pause")
            self.play_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.play_thread.start()

    def _playback_loop(self):
        delay = 1.0 / (self.processor.fps or 30.0)
        while not self.stop_flag.is_set():
            ok, frame, _ = self.processor.read_frame()
            if not ok:
                self.processor.seek(0)
                self.playing = False
                self.play_btn.config(text="▶ Play")
                break
            self.after(0, self._display_frame, frame)
            current, total = self.processor.get_progress()
            self.after(0, lambda c=current: self.progress.config(value=c))
            cv2.waitKey(1)
            self.stop_flag.wait(delay)

    def _display_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        target_w = max(self.video_label.winfo_width(), 400)
        target_h = max(self.video_label.winfo_height(), 300)
        img.thumbnail((target_w, target_h))

        self.current_photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=self.current_photo, text="")

    # ---------------- full-video analysis ----------------

    def run_full_analysis(self):
        if not self.processor:
            return
        self.stop_flag.set()
        self.playing = False
        self.play_btn.config(text="▶ Play")
        self.status_var.set("Analyzing video — this may take a moment...")
        self.analyze_btn.config(state="disabled")
        threading.Thread(target=self._analyze_worker, daemon=True).start()

    def _analyze_worker(self):
        self.processor.seek(0)
        self.analyzer = StrokeAnalyzer(fps=self.processor.fps)
        total = self.processor.frame_count or 1

        frame_idx = 0
        while True:
            ok, frame, landmarks = self.processor.read_frame()
            if not ok:
                break
            self.analyzer.add_frame(landmarks)
            frame_idx += 1
            if frame_idx % 3 == 0:
                self.after(0, self._display_frame, frame)
                self.after(0, lambda f=frame_idx: self.progress.config(value=f))
            self.after(0, lambda f=frame_idx, t=total: self.status_var.set(f"Analyzing frame {f}/{t}"))

        summary = self.analyzer.summary()
        self.after(0, self._show_report, summary)
        self.processor.seek(0)

    def _show_report(self, summary):
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        lines = [
            f"Stroke rate:        {summary['stroke_rate_spm']} strokes/min",
            f"Left stroke count:  {summary['left_stroke_count']}",
            f"Right stroke count: {summary['right_stroke_count']}",
            f"Avg left elbow:     {summary['avg_left_elbow_angle']}°",
            f"Avg right elbow:    {summary['avg_right_elbow_angle']}°",
            f"Avg body roll:      {summary['avg_body_roll_deg']}°",
            f"Symmetry score:     {summary['symmetry_score']} / 100",
            "",
            "Feedback:",
        ] + [f" • {f}" for f in summary["feedback"]]
        self.metrics_text.insert("end", "\n".join(lines))
        self.metrics_text.config(state="disabled")

        self._draw_charts(summary)
        self.last_summary = summary
        self.export_btn.config(state="normal")
        self.analyze_btn.config(state="normal")
        self.status_var.set("Analysis complete.")

    def _draw_charts(self, summary):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(3.9, 2.6), dpi=100, facecolor="#0d1b2a")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1b263b")
        ax.plot(summary["left_elbow_series"], label="Left elbow", color="#48cae4", linewidth=1.2)
        ax.plot(summary["right_elbow_series"], label="Right elbow", color="#f4a261", linewidth=1.2)
        ax.set_title("Elbow angle over time", color="#e0e1dd", fontsize=9)
        ax.tick_params(colors="#8d99ae", labelsize=7)
        ax.legend(fontsize=7, facecolor="#1b263b", labelcolor="#e0e1dd")
        for spine in ax.spines.values():
            spine.set_color("#415a77")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------- export ----------------

    def export_report(self):
        if not hasattr(self, "last_summary"):
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
            initialfile="stroke_analysis_report.txt",
        )
        if not path:
            return
        s = self.last_summary
        with open(path, "w") as f:
            f.write("SWIMMING STROKE ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n")
            f.write(f"Video: {os.path.basename(self.video_path)}\n\n")
            f.write(f"Stroke rate: {s['stroke_rate_spm']} strokes/min\n")
            f.write(f"Left stroke count: {s['left_stroke_count']}\n")
            f.write(f"Right stroke count: {s['right_stroke_count']}\n")
            f.write(f"Avg left elbow angle: {s['avg_left_elbow_angle']}\n")
            f.write(f"Avg right elbow angle: {s['avg_right_elbow_angle']}\n")
            f.write(f"Avg body roll: {s['avg_body_roll_deg']}\n")
            f.write(f"Symmetry score: {s['symmetry_score']}\n\n")
            f.write("Feedback:\n")
            for item in s["feedback"]:
                f.write(f" - {item}\n")
        messagebox.showinfo("Exported", f"Report saved to:\n{path}")

    def on_close(self):
        self.stop_flag.set()
        if self.processor:
            self.processor.release()
        self.destroy()


if __name__ == "__main__":
    app = SwimAnalyzerApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
