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
MODEL_PATH = os.path.join("distance_jump", "stages", "stage1", "models", "distance_stage1.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

model = load_model(MODEL_PATH, custom_objects={'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))})
MAX_SEQUENCE_LENGTH = model.input_shape[1]  # Adjust based on training data
NUM_FEATURES = model.input_shape[-1]  # Feature dimension based on the model

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def is_accelerating(hip_prev, hip_curr, threshold=0.01):
    """
    Determines if the athlete is accelerating based on the hip's forward movement.
    Conditions:
    - The hip should show consistent forward movement.
    - A threshold is used to detect significant changes.
    """
    return hip_curr[0] - hip_prev[0] > threshold  # Check x-axis movement

def extract_keypoints_with_acceleration(video_path):
    """
    Extract keypoints for acceleration analysis.
    """
    keypoints = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    prev_hip_position = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Extract left and right hip positions
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]

            # Average the left and right hip for overall movement tracking
            avg_hip_x = (left_hip[0] + right_hip[0]) / 2
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            current_hip = [avg_hip_x, avg_hip_y]

            # Determine acceleration based on hip movement
            accelerating = False
            if prev_hip_position is not None:
                accelerating = is_accelerating(prev_hip_position, current_hip)

            # Store acceleration status (1 if accelerating, 0 otherwise)
            keypoints.append([int(accelerating), 1])  # Include constant input for stability

            prev_hip_position = current_hip

    cap.release()
    return keypoints

def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.7:
        return 1.0
    elif prediction >= 0.45:
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

        keypoints = extract_keypoints_with_acceleration(video_path)
        if not keypoints:
            return {"video": video_path, "error": "No keypoints extracted"}

        # Pad the sequence to match model input length
        keypoints_padded = pad_sequences([keypoints], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
        keypoints_padded = keypoints_padded.reshape((1, MAX_SEQUENCE_LENGTH, NUM_FEATURES))

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
    test_video_path = os.path.join("distance_jump", "stages", "stage1", "test_videos", "1_user3.mp4")
    result = process_video(test_video_path)
    print(result)