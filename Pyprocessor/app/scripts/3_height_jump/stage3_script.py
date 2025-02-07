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
MODEL_PATH = os.path.join("height_jump", "stages", "stage3", "models", "highjump_stage3.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

model = load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))
    },
)
MAX_SEQUENCE_LENGTH = model.input_shape[1]  # Adjust based on training data
NUM_FEATURES = model.input_shape[-1]  # Left foot and right foot binary values

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to check if the knee is lifted and the arm is high
def is_knee_lift_arm_high(knee, ankle, shoulder, wrist):
    knee_x, knee_y = knee
    ankle_x, ankle_y = ankle
    shoulder_x, shoulder_y = shoulder
    wrist_x, wrist_y = wrist

    knee_lifted = knee_y < ankle_y - 0.1  # Knee should be higher than ankle
    arm_high = wrist_y < shoulder_y       # Wrist should be higher than shoulder

    return knee_lifted and arm_high

def extract_keypoints(video_path):
    """
    Extract keypoints based on knee lift and arm high analysis.
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

            # Extract keypoints for knee lift and arm high analysis
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            # Check for knee lift and arm position for both sides
            left_takeoff = is_knee_lift_arm_high(left_knee, left_ankle, left_shoulder, left_wrist)
            right_takeoff = is_knee_lift_arm_high(right_knee, right_ankle, right_shoulder, right_wrist)

            # Store the binary values (0 or 1) for analysis
            keypoints.append([int(left_takeoff), int(right_takeoff)])

    cap.release()
    return keypoints


def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.7:
        return 1.0
    elif prediction >= 0.35:
        return 0.5
    else:
        return 0.0


def process_video(video_path):
    """
    Process the video and return evaluation results.
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
    test_video_path = os.path.join("height_jump", "stages", "stage3", "test_videos", "1_user10.mp4")
    result = process_video(test_video_path)
    print(result)