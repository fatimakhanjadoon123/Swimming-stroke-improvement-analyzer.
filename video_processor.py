"""
video_processor.py

Handles video I/O and per-frame pose (skeleton) detection using MediaPipe.
Every processed frame yields:
    - the annotated BGR frame (with skeleton drawn on it)
    - a dict of key landmark coordinates (normalized 0-1) used later
      by stroke_analyzer.py to compute swimming-form metrics.
"""

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# The landmarks we actually care about for stroke analysis
KEY_POINTS = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}


class VideoProcessor:
    """Wraps an OpenCV VideoCapture + MediaPipe Pose pipeline."""

    def __init__(self, video_path, model_complexity=1):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.pose = mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def read_frame(self):
        """Reads and processes the next frame.

        Returns (success, annotated_bgr_frame, landmarks_dict_or_None)
        """
        success, frame = self.cap.read()
        if not success:
            return False, None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        landmarks = None
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            )
            landmarks = {}
            for name, idx in KEY_POINTS.items():
                lm = results.pose_landmarks.landmark[idx]
                landmarks[name] = (lm.x, lm.y, lm.visibility)

        return True, frame, landmarks

    def get_progress(self):
        current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return current, self.frame_count

    def seek(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def release(self):
        self.cap.release()
        self.pose.close()
