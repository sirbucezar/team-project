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
MODEL_PATH = os.path.join("distance_jump", "stages", "stage2", "models", "distance_stage2.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

model = load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))
    },
)
MAX_SEQUENCE_LENGTH = model.input_shape[1]  # Adjust based on training data
NUM_FEATURES = model.input_shape[-1] 

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def is_foot_on_plank_and_no_gaze(foot, plank_threshold, gaze_y, board_y):
    """
    Determines if the foot lands on the plank and the athlete is not looking at the board.
    Conditions:
    - Foot should be within plank threshold (x-axis).
    - Gaze (nose position) should not align with the board (y-axis).
    """
    foot_x, _ = foot
    not_gazing = gaze_y > board_y  # Checking if gaze is above the board
    foot_on_plank = foot_x < plank_threshold
    return foot_on_plank and not_gazing


def extract_keypoints_with_gaze_and_foot(video_path):
    """
    Extracts keypoints based on foot positions and gaze for the given video.
    """
    keypoints = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    plank_threshold = 0.5  # Modify based on expected foot x positions
    board_y_position = 0.3  # Approximate y position of the board

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Extract keypoints for left and right foot and gaze (nose position)
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x,
                    landmarks[mp_pose.PoseLandmark.NOSE].y]

            # Check foot position and gaze status
            left_foot_status = is_foot_on_plank_and_no_gaze(left_foot, plank_threshold, nose[1], board_y_position)
            right_foot_status = is_foot_on_plank_and_no_gaze(right_foot, plank_threshold, nose[1], board_y_position)

            # Store binary values (0 or 1) for both feet
            keypoints.append([int(left_foot_status), int(right_foot_status)])

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

        keypoints = extract_keypoints_with_gaze_and_foot(video_path)
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
    test_video_path = os.path.join("distance_jump", "stages", "stage2", "test_videos", "0_user9.mp4")
    result = process_video(test_video_path)
    print(result)