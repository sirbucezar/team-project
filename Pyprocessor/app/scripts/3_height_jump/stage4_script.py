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
MODEL_PATH = os.path.join("height_jump", "stages", "stage4", "models", "highjump_stage4.keras")
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

# Function to check if the athlete clears the bar with an arched back
def is_arched_back(shoulder, hip, back_mid):
    """
    Determines if the athlete clears the bar with an arched back by analyzing
    the curvature of the back (shoulder, hip, and midpoint of the back).
    Conditions:
    - The back midpoint should be lower (y-axis) than the shoulder and hip.
    - Shoulder and hip should maintain a horizontal distance.
    """
    shoulder_x, shoulder_y = shoulder
    hip_x, hip_y = hip
    back_mid_x, back_mid_y = back_mid

    arched_back = back_mid_y > shoulder_y and back_mid_y > hip_y
    return arched_back

def extract_keypoints(video_path):
    """
    Extract keypoints based on arched back analysis.
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

            # Extract keypoints for arched back analysis
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_back_mid = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                             (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y) / 2]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_back_mid = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                              (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2]

            # Check for arched back posture for both sides
            left_arch = is_arched_back(left_shoulder, left_hip, left_back_mid)
            right_arch = is_arched_back(right_shoulder, right_hip, right_back_mid)

            # Store the binary values (0 or 1) for analysis
            keypoints.append([int(left_arch), int(right_arch)])

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
    test_video_path = os.path.join("height_jump", "stages", "stage4", "test_videos", "1_user12.mp4")
    result = process_video(test_video_path)
    print(result)
