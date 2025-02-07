import os
import cv2
import json
import numpy as np
import mediapipe as mp
import logging
from tensorflow.keras.models import load_model
from app.utils import get_local_path

logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join("app", "models", "2_shotput", "shotput_stage5.keras")
MAX_SEQUENCE_LENGTH = 100  # Based on original working script

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def extract_keypoints(video_path):
    """
    Extract keypoints and detect the release frame.
    For each frame, store wrist, neck, and shoulder.
    Returns a tuple (keypoints, release_frame) where keypoints is a list
    and release_frame is the index at which a significant jump in distance occurs.
    """
    keypoints = []
    distances = []
    release_frame = None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            neck = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            
            keypoints.append({"wrist": wrist, "neck": neck, "shoulder": shoulder})
            distances.append(calculate_distance(wrist, neck))
    
    cap.release()

    # Detect release frame as the first frame where distance increases significantly
    for i in range(1, len(distances)):
        if distances[i] > distances[i - 1] * 1.5:
            release_frame = i
            break

    if release_frame is None:
        release_frame = len(distances) - 1

    return keypoints, release_frame

def extract_features(keypoints, release_frame):
    """
    From the extracted keypoints, compute features.
    For each frame, compute:
      - The distance between wrist and neck.
      - The release angle at the detected release frame.
    """
    if release_frame is None or release_frame <= 0:
        return np.array([])
    
    features = []
    for i in range(len(keypoints)):
        wrist = keypoints[i]["wrist"]
        neck = keypoints[i]["neck"]
        feature_dict = {"shot_neck_distance": calculate_distance(wrist, neck)}

        if i == release_frame and i > 0:
            prev_wrist = keypoints[i-1]["wrist"]
            release_angle = np.degrees(np.arctan2(wrist[1] - prev_wrist[1], wrist[0] - prev_wrist[0]))
            feature_dict["release_angle"] = release_angle
        else:
            feature_dict["release_angle"] = 0

        features.append(list(feature_dict.values()))

    return np.array(features, dtype=np.float32)

def classify_prediction(predictions):
    """Map prediction vector to a score using argmax."""
    class_map = {0: 0, 1: 0.5, 2: 1}
    predicted_class = np.argmax(predictions)
    return class_map[predicted_class]

def main(video_path):
    """Process video for stage 5 analysis and return results."""
    local_path = None
    try:
        local_path = get_local_path(video_path)
        keypoints, release_frame = extract_keypoints(local_path)

        if not keypoints:
            return {"video": video_path, "error": "No keypoints extracted"}
        
        features = extract_features(keypoints, release_frame)
        if features.size == 0:
            return {"video": video_path, "error": "No features extracted"}

        # Reshape features correctly for model input
        features = features.reshape(1, features.shape[0], features.shape[1])

        # Load the model
        model = load_model(MODEL_PATH)

        # Predict and classify
        predictions = model.predict(features)
        score = classify_prediction(predictions[0])

        return {
            "video": video_path,
            "predicted_scores": predictions.tolist(),
            "classified_score": float(score),
            "release_frame": release_frame
        }

    except Exception as e:
        logger.error(f"Error processing video in stage5: {str(e)}")
        return {"video": video_path, "error": str(e)}
    finally:
        if local_path and os.path.exists(local_path):
            os.unlink(local_path)

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_videos", "stage5.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))