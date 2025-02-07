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
MODEL_PATH = os.path.join("hurdling", "stages", "stage1", "models", "hurdles_stage1.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

model = load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))
    },
)
MAX_SEQUENCE_LENGTH = model.input_shape[1]
NUM_FEATURES = model.input_shape[-1]  # Number of features (binary or step-related values)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to count steps based on hip movement
def count_steps(hip_positions):
    """
    Counts the number of steps by tracking the hip movement along the x-axis.
    A step is detected when the hip changes direction.
    """
    step_count = 0
    previous_x = None
    direction = None

    for hip_x in hip_positions:
        if previous_x is not None:
            if hip_x > previous_x and direction != "right":
                step_count += 1
                direction = "right"
            elif hip_x < previous_x and direction != "left":
                step_count += 1
                direction = "left"
        previous_x = hip_x

    return step_count

# Function to extract keypoints and calculate steps
def extract_keypoints(video_path):
    """
    Extract keypoints and count steps for a given video.
    """
    keypoints = []
    right_hip_positions = []
    left_hip_positions = []

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

            # Extract positions for both hips
            right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
            left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x

            right_hip_positions.append(right_hip_x)
            left_hip_positions.append(left_hip_x)

    cap.release()

    # Count steps for both hips
    num_right_steps = count_steps(right_hip_positions)
    num_left_steps = count_steps(left_hip_positions)

    keypoints.append([num_right_steps, num_left_steps])
    return keypoints

# Function to classify scores
def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.7:
        return 1.0
    elif prediction >= 0.35:
        return 0.5
    else:
        return 0.0

# Main function to process the video
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
        keypoints_padded = pad_sequences(keypoints, maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
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
    test_video_path = os.path.join("hurdling", "stages", "stage1", "test_videos", "1_user1.mp4")
    result = process_video(test_video_path)
    print(result)