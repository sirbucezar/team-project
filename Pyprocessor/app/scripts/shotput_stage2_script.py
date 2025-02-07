import os
import cv2
import json
import numpy as np
import mediapipe as mp
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from app.utils import get_local_path

logger = logging.getLogger(__name__)

# Correct model path handling to avoid misplacement issues
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "2_shotput", "shotput_stage2.keras")
MAX_SEQUENCE_LENGTH = 76  # Defined based on the model's expected input length

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """Calculate the angle between three points with clipping for numerical stability."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def classify_score(prediction):
    """Classify prediction into 0, 0.5, or 1 based on defined thresholds."""
    return 1.0 if prediction >= 0.90 else 0.5 if prediction >= 0.75 else 0.0

def extract_features_from_video(video_path):
    """
    Extract features (angle and distance) from the video.
    Returns a NumPy array with two columns: angles and distances.
    """
    angles, distances = [], []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                angle_val = calculate_angle(right_hip, right_knee, right_ankle)
                dist_val = calculate_distance(right_hip, right_knee)

                angles.append(angle_val)
                distances.append(dist_val)

    finally:
        cap.release()

    if not angles or not distances:
        logger.error("No valid keypoints detected from the video.")
        return np.array([])

    features = np.stack([angles, distances], axis=1)
    return features

def main(video_path):
    """Process video for stage 2 analysis and return results."""
    local_path = None
    try:
        logger.info(f"Processing video: {video_path}")
        local_path = get_local_path(video_path)

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local video file not found: {local_path}")

        features = extract_features_from_video(local_path)
        
        if features.size == 0:
            logger.error("No features extracted from the video.")
            return {"video": video_path, "error": "No features extracted"}
        
        # Pad or truncate feature sequences to fit model input shape
        if features.shape[0] < MAX_SEQUENCE_LENGTH:
            padding = np.zeros((MAX_SEQUENCE_LENGTH - features.shape[0], 2))
            features_padded = np.vstack((features, padding))
        else:
            features_padded = features[:MAX_SEQUENCE_LENGTH, :]

        features_padded = np.expand_dims(features_padded, axis=[0])  # Add batch dimension

        # Load the model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        model = load_model(MODEL_PATH)
        prediction = model.predict(features_padded)[0][0]
        score = classify_score(prediction)

        logger.info(f"Prediction: {prediction:.2f}, Classified Score: {score}")

        return {
            "video": video_path,
            "predicted_score": float(prediction),
            "classified_score": score
        }

    except Exception as e:
        logger.error(f"Error processing video in stage2: {str(e)}")
        return {"video": video_path, "error": str(e)}
    finally:
        if local_path and os.path.exists(local_path):
            os.unlink(local_path)
            logger.info(f"Deleted temporary file: {local_path}")

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(__file__), "..", "test_videos", "stage2.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))