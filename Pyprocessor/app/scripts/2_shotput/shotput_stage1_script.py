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

# Constants
MODEL_SUBPATH = os.path.join( "models", "2_shotput", "shotput_stage1.keras")
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
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def extract_keypoints(video_path):
    """
    Extract keypoints (here, the left leg angle) from a video file.
    Returns a list of angles.
    """
    keypoints = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    while True:
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
            angle_left_leg = calculate_angle(left_hip, left_knee, left_ankle)
            keypoints.append(angle_left_leg)
    cap.release()
    return keypoints

def classify_score(prediction):
    """Convert a prediction value into a score of 0, 0.5, or 1."""
    if prediction >= 0.85:
        return 1.0
    elif prediction >= 0.70:
        return 0.5
    else:
        return 0.0

def main(video_path):
    """Main function to process a video and return analysis results."""
    local_path = None
    try:
        local_path = get_local_path(video_path)
        keypoints = extract_keypoints(local_path)
        if not keypoints:
            return {"video": video_path, "error": "No keypoints extracted"}
        
        # Build the absolute model path relative to the package root.
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, MODEL_SUBPATH)
        
        # Pad the sequence so that it matches the input requirements of the model.
        keypoints_padded = pad_sequences([keypoints], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
        keypoints_padded = keypoints_padded[..., np.newaxis]  # Expand dims to (1, seq_length, 1)
        
        # Load the model with the custom loss.
        model = load_model(model_path, custom_objects={"weighted_mse": weighted_mse})
        prediction = model.predict(keypoints_padded)[0][0]
        score = classify_score(prediction)
        
        return { 
            "video": video_path,
            "predicted_score": float(prediction),
            "classified_score": score
        }
    except Exception as e:
        logger.error(f"Error processing video in stage1: {str(e)}")
        return {"video": video_path, "error": str(e)}
    finally:
        # Clean up the temporary file if the video was downloaded.
        if video_path.startswith(("http://", "https://")) and local_path and os.path.exists(local_path):
            try:
                os.unlink(local_path)
            except Exception as cleanup_e:
                logger.error(f"Error cleaning up temporary file: {str(cleanup_e)}")

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_videos", "stage1.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))