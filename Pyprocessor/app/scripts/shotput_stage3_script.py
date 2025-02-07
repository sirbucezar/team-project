import os
import cv2
import json
import numpy as np
import mediapipe as mp
import logging
from tensorflow.keras.models import load_model
from app.utils import get_local_path

logger = logging.getLogger(__name__)

# Correcting model path handling to avoid misplacement issues
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "2_shotput", "shotput_stage3.keras")
MAX_SEQUENCE_LENGTH = 100  # Defined based on the model's expected input length

# Initialize MediaPipe Pose with tracking confidence parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculate the angle between three points with clipping for numerical stability."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_velocity(coord1, coord2, fps=30):
    """Calculate velocity between two coordinate points given fps."""
    dx, dy = coord2[0] - coord1[0], coord2[1] - coord1[1]
    distance = np.sqrt(dx**2 + dy**2)
    return distance * fps

def extract_keypoints(video_path):
    """Extract left and right leg keypoints from the video."""
    keypoints = []
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
                left_leg = {
                    "hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                    "knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                    "ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                }
                right_leg = {
                    "hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
                    "knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
                    "ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y],
                }
                keypoints.append({"left_leg": left_leg, "right_leg": right_leg})
    finally:
        cap.release()

    if not keypoints:
        logger.error("No valid keypoints detected from the video.")
    
    return keypoints

def extract_features(keypoints):
    """
    Extract features from keypoints.
    For each frame (starting at frame 1), calculate:
      - The velocity of the left ankle.
      - The knee angle from the right leg.
      - The distance between left and right ankles.
    Returns an array of features.
    """
    features = []
    for i in range(1, len(keypoints)):
        prev_left_ankle = keypoints[i - 1]["left_leg"]["ankle"]
        curr_left_ankle = keypoints[i]["left_leg"]["ankle"]
        left_velocity = calculate_velocity(prev_left_ankle, curr_left_ankle)

        right_leg = keypoints[i]["right_leg"]
        knee_angle = calculate_angle(right_leg["hip"], right_leg["knee"], right_leg["ankle"])

        left_ankle, right_ankle = keypoints[i]["left_leg"]["ankle"], keypoints[i]["right_leg"]["ankle"]
        ankle_distance = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))

        features.append([left_velocity, knee_angle, ankle_distance])

    if not features:
        logger.error("No features extracted from the keypoints.")
        return np.array([])

    return np.array(features)

def classify_score(predictions):
    """Map prediction to class labels (0, 0.5, 1)."""
    score_mapping = {0: 0, 1: 0.5, 2: 1}
    return score_mapping[np.argmax(predictions)]

def main(video_path):
    """Process video for stage 3 analysis and return results."""
    local_path = None
    try:
        logger.info(f"Processing video: {video_path}")
        local_path = get_local_path(video_path)

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local video file not found: {local_path}")

        keypoints = extract_keypoints(local_path)
        if not keypoints:
            logger.error("No keypoints extracted.")
            return {"video": video_path, "error": "No keypoints extracted"}

        features = extract_features(keypoints)
        if features.size == 0:
            logger.error("No features extracted from keypoints.")
            return {"video": video_path, "error": "No features extracted"}

        # Ensure padding
        if features.shape[0] < MAX_SEQUENCE_LENGTH:
            padding = np.zeros((MAX_SEQUENCE_LENGTH - features.shape[0], 3))
            features_padded = np.vstack((features, padding))
        else:
            features_padded = features[:MAX_SEQUENCE_LENGTH, :]

        features_padded = np.expand_dims(features_padded, axis=[0])  # Add batch dimension

        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        model = load_model(MODEL_PATH)

        # Predict and classify
        predictions = model.predict(features_padded)
        classified_score = classify_score(predictions)

        logger.info(f"Predicted scores: {predictions.tolist()}, Classified Score: {classified_score}")

        return {
            "video": video_path,
            "predicted_scores": predictions.tolist(),
            "classified_score": float(classified_score)
        }

    except Exception as e:
        logger.error(f"Error processing video in stage3: {str(e)}")
        return {"video": video_path, "error": str(e)}
    finally:
        if local_path and os.path.exists(local_path):
            os.unlink(local_path)
            logger.info(f"Deleted temporary file: {local_path}")

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(__file__), "..", "test_videos", "stage3.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))