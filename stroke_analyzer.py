"""
stroke_analyzer.py

Turns a sequence of per-frame landmark dicts (from video_processor.py)
into swimming-form metrics:
    - stroke count / stroke rate (strokes per minute)
    - elbow (pull) angle, left vs right, over time
    - body rotation ("roll") angle, shoulders vs hips
    - left/right symmetry score
    - simple coaching-style feedback text
"""

import numpy as np


def _angle_between(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees. Points are (x, y)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) or 1e-9
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class StrokeAnalyzer:
    def __init__(self, fps):
        self.fps = fps
        self.frames = []  # list of landmark dicts (None for missed detections)

    def add_frame(self, landmarks):
        self.frames.append(landmarks)

    # ---------- core metric computations ----------

    def _series(self, key, component=1):
        """Extract a time series for a landmark's x(0)/y(1) coordinate,
        skipping frames with no detection."""
        out = []
        for lm in self.frames:
            if lm and key in lm:
                out.append(lm[key][component])
            else:
                out.append(np.nan)
        return np.array(out)

    def elbow_angles(self):
        left, right = [], []
        for lm in self.frames:
            if not lm:
                left.append(np.nan)
                right.append(np.nan)
                continue
            left.append(_angle_between(lm["left_shoulder"][:2], lm["left_elbow"][:2], lm["left_wrist"][:2]))
            right.append(_angle_between(lm["right_shoulder"][:2], lm["right_elbow"][:2], lm["right_wrist"][:2]))
        return np.array(left), np.array(right)

    def body_roll_angles(self):
        """Approximate body roll using the angle of the shoulder line vs horizontal."""
        rolls = []
        for lm in self.frames:
            if not lm:
                rolls.append(np.nan)
                continue
            ls = np.array(lm["left_shoulder"][:2])
            rs = np.array(lm["right_shoulder"][:2])
            dx, dy = (rs - ls)
            angle = np.degrees(np.arctan2(dy, dx))
            rolls.append(angle)
        return np.array(rolls)

    def count_strokes(self, wrist="right_wrist"):
        """Counts strokes by detecting peaks in the wrist's vertical (y) motion.
        A stroke = one full recovery cycle (one local minimum in y, since
        image-y grows downward -> a minimum = wrist near the surface/top)."""
        y = self._series(wrist, component=1)
        valid_idx = np.where(~np.isnan(y))[0]
        if len(valid_idx) < 5:
            return 0, []

        y_clean = y[valid_idx]
        # light smoothing to avoid counting jitter as strokes
        kernel = np.ones(5) / 5
        y_smooth = np.convolve(y_clean, kernel, mode="same")

        peaks = []
        threshold = 0.01  # normalized-coordinate noise threshold
        for i in range(2, len(y_smooth) - 2):
            window = y_smooth[i - 2:i + 3]
            if y_smooth[i] == window.min() and (window.max() - window.min()) > threshold:
                if not peaks or (valid_idx[i] - peaks[-1]) > self.fps * 0.3:
                    peaks.append(valid_idx[i])

        return len(peaks), peaks

    def stroke_rate(self):
        """Strokes per minute, averaged across both arms."""
        total_frames = len(self.frames)
        duration_min = (total_frames / self.fps) / 60.0 if self.fps else 0
        if duration_min <= 0:
            return 0.0, 0, 0

        left_count, _ = self.count_strokes("left_wrist")
        right_count, _ = self.count_strokes("right_wrist")
        avg_count = (left_count + right_count) / 2.0
        return avg_count / duration_min, left_count, right_count

    def symmetry_score(self):
        """0-100 score: how similar left vs right elbow angle patterns are."""
        left, right = self.elbow_angles()
        mask = ~np.isnan(left) & ~np.isnan(right)
        if mask.sum() < 5:
            return None
        diff = np.abs(left[mask] - right[mask])
        mean_diff = np.nanmean(diff)
        score = max(0.0, 100.0 - mean_diff)  # heuristic: smaller diff => higher symmetry
        return round(score, 1)

    def summary(self):
        """Returns a dict with all computed metrics plus plain-language feedback."""
        rate, left_n, right_n = self.stroke_rate()
        left_angle, right_angle = self.elbow_angles()
        roll = self.body_roll_angles()
        sym = self.symmetry_score()

        avg_left_elbow = float(np.nanmean(left_angle)) if np.any(~np.isnan(left_angle)) else None
        avg_right_elbow = float(np.nanmean(right_angle)) if np.any(~np.isnan(right_angle)) else None
        avg_roll = float(np.nanmean(np.abs(roll))) if np.any(~np.isnan(roll)) else None

        feedback = []
        if sym is not None and sym < 80:
            feedback.append("Noticeable left/right arm asymmetry detected — focus on even pull timing.")
        if avg_left_elbow is not None and avg_right_elbow is not None:
            if min(avg_left_elbow, avg_right_elbow) < 90:
                feedback.append("One or both elbows drop below a 90° catch angle — work on a high-elbow catch.")
        if rate and (rate < 40 or rate > 70):
            feedback.append(f"Stroke rate of {rate:.1f} strokes/min is outside the typical 40–70 range for freestyle.")
        if not feedback:
            feedback.append("Stroke mechanics look within a solid, balanced range. Keep it up!")

        return {
            "stroke_rate_spm": round(rate, 1),
            "left_stroke_count": left_n,
            "right_stroke_count": right_n,
            "avg_left_elbow_angle": round(avg_left_elbow, 1) if avg_left_elbow else None,
            "avg_right_elbow_angle": round(avg_right_elbow, 1) if avg_right_elbow else None,
            "avg_body_roll_deg": round(avg_roll, 1) if avg_roll else None,
            "symmetry_score": sym,
            "feedback": feedback,
            "left_elbow_series": left_angle,
            "right_elbow_series": right_angle,
            "roll_series": roll,
        }
