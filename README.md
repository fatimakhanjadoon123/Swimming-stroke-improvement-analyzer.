# 🏊 Swimming Stroke Improvement Analyzer

**Category:** Video analysis · **Tech stack:** Tkinter, OpenCV, MediaPipe · **Difficulty:** Advanced

> Project brief (from spec): *"Analyze form from race videos."*

A desktop app that loads a swimmer's race video, overlays a live pose
skeleton on top of it, and generates a stroke-analysis report covering:

- **Stroke rate** (strokes per minute, left & right arm)
- **Elbow / catch angle** over time (left vs right)
- **Body roll** (rotation of the shoulder line)
- **Left/right symmetry score**
- Plain-language coaching feedback based on the numbers above
- A downloadable `.txt` report

---

## 1. Requirements

- Python 3.9 – 3.11 (MediaPipe does not yet support the very latest Python versions)
- A webcam is **not** required — this works from any video file (mp4, mov, avi, mkv)

## 2. Setup in VS Code

1. Open this folder (`swim_stroke_analyzer/`) in VS Code.
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate it:

   - **Windows:** `venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`

4. In VS Code, select the venv's interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `venv`.

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 3. Run the app

```bash
python main.py
```

Or press `F5` in VS Code (with the venv interpreter selected).

## 4. Using the app

1. Click **Load Race Video** and pick a swimming clip (side-on shots of
   freestyle/front crawl work best for pose detection).
2. Click **▶ Play** to preview it with the live skeleton overlay.
3. Click **Analyze Full Video** to process every frame and compute the
   stroke metrics — this can take a little while depending on video length.
4. Review the report and elbow-angle chart in the right-hand panel.
5. Click **Export Report** to save a `.txt` summary you can share with a
   swimmer or coach.

## 5. Project structure

```
swim_stroke_analyzer/
├── main.py              # Tkinter GUI, video playback & orchestration
├── video_processor.py   # OpenCV + MediaPipe pose detection per frame
├── stroke_analyzer.py   # Stroke rate, angles, symmetry, feedback logic
├── requirements.txt
└── README.md
```

## 6. Notes & tips

- Video should ideally show the swimmer **from the side**, fully in
  frame, for the most reliable pose detection.
- Analysis quality depends on video resolution/lighting — MediaPipe
  Pose can lose tracking on very fast, blurry, or heavily splashed
  footage; missed frames are automatically skipped in the metrics.
- Feel free to tweak the thresholds in `stroke_analyzer.py`
  (e.g. `threshold` in `count_strokes`, or the feedback rules in
  `summary()`) to better match a specific stroke style or camera angle.
