import logging
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join("disc_throwing", "stages", "stage1", "models", "discus_stage1.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

model = load_model(MODEL_PATH, custom_objects={'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))})
MAX_SEQUENCE_LENGTH = model.input_shape[1]  # Adjust based on training data

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def is_arm_swinging_back(shoulder, elbow, wrist):
    """
    Determines if the throwing arm swings back after initial motion by analyzing
    the relative positions of the shoulder, elbow, and wrist.
    """
    return wrist[0] < elbow[0] and elbow[0] < shoulder[0]

def extract_keypoints(video_path):
    """
    Extract keypoints for throwing arm swing analysis.
    """
    keypoints = []
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

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            left_swing = is_arm_swinging_back(left_shoulder, left_elbow, left_wrist)
            right_swing = is_arm_swinging_back(right_shoulder, right_elbow, right_wrist)

            keypoints.append([int(left_swing), int(right_swing)])

    cap.release()
    return keypoints

def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.65:
        return 1.0
    elif prediction >= 0.4:
        return 0.5
    else:
        return 0.0

def process_video(video_path):
    """
    Process the discus throw video and return evaluation results.
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Error: Video file not found at {video_path}")

        keypoints = extract_keypoints(video_path)
        if not keypoints:
            return {"video": video_path, "error": "No keypoints extracted"}

        # Pad the sequence to match model input length
        keypoints_padded = pad_sequences([keypoints], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
        keypoints_padded = keypoints_padded.reshape((1, MAX_SEQUENCE_LENGTH, 2))

        # Make predictions
        predictions = model.predict(keypoints_padded)

        classified_score = classify_score(float(predictions[0][0]))
        return {
            "video": video_path,
            "predicted_score": float(predictions[0][0]),
            "classified_score": classified_score
        }
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {"video": video_path, "error": str(e)}

if __name__ == "__main__":
    test_video_path = os.path.join("disc_throwing", "stages", "stage1", "test_videos", "1_user10.mp4")
    result = process_video(test_video_path)
    print(result)
