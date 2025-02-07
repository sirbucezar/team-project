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
MODEL_PATH = os.path.join("disc_throwing", "stages", "stage5", "models", "discus_stage5.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

model = load_model(MODEL_PATH, custom_objects={'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))})
MAX_SEQUENCE_LENGTH = model.input_shape[1]  # Adjust based on training data
NUM_FEATURES = model.input_shape[-1]  # Feature dimension based on the model

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def is_discus_released(index_finger, wrist):
    """
    Determines if the discus is released via the index finger.
    Conditions:
    - The index finger should extend beyond the wrist in the x-direction.
    - The vertical displacement should be minimal.
    """
    index_x, index_y = index_finger
    wrist_x, wrist_y = wrist
    return index_x > wrist_x and abs(index_y - wrist_y) < 0.05

def extract_keypoints(video_path):
    """
    Extract keypoints for discus release and other analysis.
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
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Extract keypoints for discus release analysis
            left_index_finger = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            right_index_finger = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            # Check for discus release on both hands
            left_release = is_discus_released(left_index_finger, left_wrist)
            right_release = is_discus_released(right_index_finger, right_wrist)

            # Use only one feature (left or right release) or both as needed
            keypoints.append([int(left_release)])

    cap.release()
    return keypoints

def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.7:
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
    test_video_path = os.path.join("disc_throwing", "stages", "stage5", "test_videos", "1_user2.mp4")
    result = process_video(test_video_path)
    print(result)