import logging
import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from app.utils import get_local_path

logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join("app", "models", "2_shotput", "shotput_stage3.keras")
MAX_SEQUENCE_LENGTH = 100  # Based on original working script

# Initialize MediaPipe Pose with tracking confidence parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculate the angle between three points with clipping for numerical stability."""
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_velocity(coord1, coord2, fps=30):
    """Calculate velocity between two coordinate points given fps."""
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    distance = np.sqrt(dx**2 + dy**2)
    return distance * fps

def extract_keypoints(video_path):
    """Extract left and right leg keypoints from the video."""
    keypoints = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    
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
    cap.release()
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
        prev_left_ankle = keypoints[i-1]["left_leg"]["ankle"]
        curr_left_ankle = keypoints[i]["left_leg"]["ankle"]
        left_velocity = calculate_velocity(prev_left_ankle, curr_left_ankle)
        
        right_leg = keypoints[i]["right_leg"]
        knee_angle = calculate_angle(right_leg["hip"], right_leg["knee"], right_leg["ankle"])
        
        left_ankle = keypoints[i]["left_leg"]["ankle"]
        right_ankle = keypoints[i]["right_leg"]["ankle"]
        ankle_distance = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))
        
        features.append([left_velocity, knee_angle, ankle_distance])
    return np.array(features)

def classify_score(predictions):
    """Map prediction to class labels (0, 0.5, 1)."""
    score_mapping = {0: 0, 1: 0.5, 2: 1}
    return score_mapping[np.argmax(predictions)]

def main(video_path):
    """Process video for stage 3 analysis and return results."""
    local_path = None
    try:
        local_path = get_local_path(video_path)
        keypoints = extract_keypoints(local_path)
        
        if not keypoints:
            return {"video": video_path, "error": "No keypoints extracted"}
        
        features = extract_features(keypoints)
        if features.size == 0:
            return {"video": video_path, "error": "No features extracted"}

        # Reshape features to match model input
        features = features.reshape(1, features.shape[0], features.shape[1])

        # Load the model
        model = load_model(MODEL_PATH)

        # Predict and classify
        predictions = model.predict(features)
        classified_score = classify_score(predictions)

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

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_videos", "stage3.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))