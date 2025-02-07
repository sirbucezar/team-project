import os
import numpy as np
import logging
import cv2
import mediapipe as mp
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
from app.utils import get_local_path

logger = logging.getLogger(__name__)

# Corrected model path handling
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "2_shotput", "shotput_stage1.keras")
MAX_SEQUENCE_LENGTH = 170

@register_keras_serializable()
def weighted_mse(y_true, y_pred):
    """Weighted Mean Squared Error to prioritize true negatives."""
    weights = K.switch(y_true < 0.70, 2.0, 1.0)  # Weight true negatives higher
    return K.mean(weights * K.square(y_true - y_pred))

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_keypoints(video_path):
    """Extract keypoints (left leg angle) from a video file."""
    keypoints = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            keypoints.append(calculate_angle(left_hip, left_knee, left_ankle))

    cap.release()
    if not keypoints:
        raise ValueError("No keypoints extracted from video.")
    return keypoints

def classify_score(prediction):
    """Convert a prediction value into a score of 0, 0.5, or 1."""
    return 1.0 if prediction >= 0.85 else 0.5 if prediction >= 0.70 else 0.0

def main(video_path):
    """Main function to process a video and return analysis results."""
    local_path = None
    try:
        logger.info(f"Processing video: {video_path}")
        local_path = get_local_path(video_path)

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local video file not found: {local_path}")

        keypoints = extract_keypoints(local_path)
        
        # Pad the sequence to match the input requirements of the model.
        keypoints_padded = pad_sequences([keypoints], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
        keypoints_padded = keypoints_padded[..., np.newaxis]  # Expand dims to (1, seq_length, 1)

        # Load the model with the custom loss.
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        model = load_model(MODEL_PATH, custom_objects={"weighted_mse": weighted_mse})
        prediction = model.predict(keypoints_padded)[0][0]
        score = classify_score(prediction)

        logger.info(f"Prediction: {prediction:.2f}, Classified Score: {score}")

        return { 
            "video": video_path,
            "predicted_score": float(prediction),
            "classified_score": score
        }
    except Exception as e:
        logger.error(f"Error processing video in stage1: {str(e)}")
        return {"video": video_path, "error": str(e)}
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Deleted temporary file: {local_path}")

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(__file__), "..", "test_videos", "stage1.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))