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
MODEL_PATH = os.path.join("disc_throwing", "stages", "stage4", "models", "discus_stage4.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

model = load_model(MODEL_PATH, custom_objects={'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))})
MAX_SEQUENCE_LENGTH = model.input_shape[1]  # Adjust based on training data

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def is_flat_pivot(ball_of_foot, heel):
    """
    Determines if the pivot is performed flat by analyzing the relative position 
    of the ball of the foot and the heel.
    Conditions:
    - The ball of the foot and heel should have minimal vertical displacement.
    - The foot should maintain a stable position on the ground.
    """
    ball_x, ball_y = ball_of_foot
    heel_x, heel_y = heel

    # Flat pivot condition: minimal vertical displacement between ball and heel
    pivot_detected = abs(ball_y - heel_y) < 0.02  # Small threshold for stability
    return pivot_detected

def extract_keypoints(video_path):
    """
    Extract keypoints for pivot analysis.
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

            left_ball_of_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]

            right_ball_of_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y]

            left_pivot = is_flat_pivot(left_ball_of_foot, left_heel)
            right_pivot = is_flat_pivot(right_ball_of_foot, right_heel)

            keypoints.append([int(left_pivot), int(right_pivot)])

    cap.release()
    return keypoints

def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.65:
        return 1.0
    elif prediction >= 0.35:
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

        # Ensure padding does not exceed actual extracted keypoints
        actual_seq_length = len(keypoints)
        padded_length = min(MAX_SEQUENCE_LENGTH, actual_seq_length)
        
        # Pad and reshape to match model input
        keypoints_padded = pad_sequences([keypoints], maxlen=padded_length, padding='post', dtype='float32')
        expected_features = model.input_shape[2]  # Get expected number of features per timestep
        
        # If the extracted sequence has fewer features, reduce dimensions
        if keypoints_padded.shape[-1] != expected_features:
            keypoints_padded = np.mean(keypoints_padded, axis=-1, keepdims=True)

        keypoints_padded = keypoints_padded.reshape((1, padded_length, expected_features))

        if keypoints_padded.shape[1] != MAX_SEQUENCE_LENGTH:
            logger.warning(f"Reshaped keypoints to match expected length. Expected: {MAX_SEQUENCE_LENGTH}, Got: {padded_length}")

        # Predict using the model
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
    test_video_path = os.path.join("disc_throwing", "stages", "stage4", "test_videos", "1_user10.mp4")
    result = process_video(test_video_path)
    print(result)
