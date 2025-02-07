import logging
import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from app.utils import get_local_path

logger = logging.getLogger(__name__)

# Correcting model path handling to avoid misplacement issues
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "2_shotput", "shotput_stage4.keras")
MAX_SEQUENCE_LENGTH = 120  # Expected input length for the model

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(video_path):
    """
    Extract keypoints for stage 4 analysis.
    For each frame, extract:
      - push_leg (right hip, knee, ankle)
      - torso (left and right shoulders)
      - arms (right elbow and wrist)
    Returns a list of dictionaries containing keypoint coordinates.
    """
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
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints.append({
                    "push_leg": {
                        "hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
                        "knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
                        "ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                    },
                    "torso": {
                        "right_shoulder": [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                        "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    },
                    "arms": {
                        "right_elbow": [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                        "right_wrist": [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                    }
                })
    finally:
        cap.release()

    if not keypoints:
        logger.error("No valid keypoints detected from the video.")

    return keypoints

def extract_features(keypoints):
    """
    Extract features from the extracted keypoints.
    For each frame, calculate:
      - push_leg velocity (x and y differences)
      - knee to ankle distance
      - shoulder angle (left to right shoulder)
      - right arm angle (elbow to wrist)
    Returns an array of feature values.
    """
    features = []
    for i in range(1, len(keypoints)):
        previous = keypoints[i - 1]["push_leg"]["hip"]
        current = keypoints[i]["push_leg"]["hip"]
        velocity = [current[0] - previous[0], current[1] - previous[1]]

        knee = keypoints[i]["push_leg"]["knee"]
        ankle = keypoints[i]["push_leg"]["ankle"]
        knee_ankle_dist = np.linalg.norm(np.array(knee) - np.array(ankle))

        right_shoulder = keypoints[i]["torso"]["right_shoulder"]
        left_shoulder = keypoints[i]["torso"]["left_shoulder"]
        shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1],
                                    right_shoulder[0] - left_shoulder[0])

        right_elbow = keypoints[i]["arms"]["right_elbow"]
        right_wrist = keypoints[i]["arms"]["right_wrist"]
        right_arm_angle = np.arctan2(right_wrist[1] - right_elbow[1],
                                     right_wrist[0] - right_elbow[0])

        features.append(velocity + [knee_ankle_dist, shoulder_angle, right_arm_angle])

    if not features:
        logger.error("No features extracted from the keypoints.")
        return np.array([])

    return np.array(features)

def classify_prediction(predictions):
    """Map the prediction to a class label."""
    class_map = {0: 0, 1: 0.5, 2: 1}
    predicted_class = np.argmax(predictions)
    return class_map[predicted_class]

def main(video_path):
    """Process video for stage 4 analysis and return results."""
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

        # Pad sequence to match model input requirements
        if features.shape[0] < MAX_SEQUENCE_LENGTH:
            padding = np.zeros((MAX_SEQUENCE_LENGTH - features.shape[0], 5))
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
        score = classify_prediction(predictions[0])

        logger.info(f"Predicted scores: {predictions.tolist()}, Classified Score: {score}")

        return {
            "video": video_path,
            "predicted_scores": predictions.tolist(),
            "classified_score": float(score)
        }

    except Exception as e:
        logger.error(f"Error processing video in stage4: {str(e)}")
        return {"video": video_path, "error": str(e)}
    finally:
        if local_path and os.path.exists(local_path):
            os.unlink(local_path)
            logger.info(f"Deleted temporary file: {local_path}")

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(__file__), "..", "test_videos", "stage4.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))