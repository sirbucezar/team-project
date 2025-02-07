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
MODEL_PATH = os.path.join("distance_jump", "stages", "stage4", "models", "distance_stage4.keras")
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


def is_knight_stance_maintained(foot, knee, hip, shoulder):
    """
    Determines if the knight stance is maintained by analyzing the alignment of 
    foot, knee, hip, and shoulder.
    Conditions:
    - Knee should be slightly ahead of the foot.
    - Hip should be aligned vertically with the knee.
    - Shoulder should be above the hip.
    """
    foot_x, foot_y = foot
    knee_x, knee_y = knee
    hip_x, hip_y = hip
    shoulder_x, shoulder_y = shoulder

    knee_ahead = knee_x > foot_x  # Knee slightly ahead of the foot
    hip_alignment = abs(hip_x - knee_x) < 0.05  # Hip aligned vertically with knee
    shoulder_above = shoulder_y < hip_y  # Shoulder above the hip

    return knee_ahead and hip_alignment and shoulder_above


def extract_keypoints_with_posture(video_path):
    """
    Extracts keypoints based on foot positions and posture for the given video.
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

            # Extract keypoints for left and right foot, knee, hip, and shoulder
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]

            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

            # Check knight stance for both sides
            left_knight_stance = is_knight_stance_maintained(left_foot, left_knee, left_hip, left_shoulder)
            right_knight_stance = is_knight_stance_maintained(right_foot, right_knee, right_hip, right_shoulder)

            # Store binary values (0 or 1) for both sides
            keypoints.append([int(left_knight_stance), int(right_knight_stance)])

    cap.release()
    return keypoints


def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.8:
        return 1.0
    elif prediction >= 0.45:
        return 0.5
    else:
        return 0.0


def process_video(video_path):
    """
    Process the knight stance video and return evaluation results.
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Error: Video file not found at {video_path}")

        keypoints = extract_keypoints_with_posture(video_path)
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
    test_video_path = os.path.join("distance_jump", "stages", "stage4", "test_videos", "1_user9.mp4")
    result = process_video(test_video_path)
    print(result)
