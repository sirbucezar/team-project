import logging
import os
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join("sprint_technique", "stages", "stage4", "models", "sprint_stage4.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

# Load the model with custom loss function
model = load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))
    },
)
MAX_SEQUENCE_LENGTH = model.input_shape[1]
NUM_FEATURES = model.input_shape[-1]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to determine if the arm is at 90° and actively moving
def is_arm_at_90_and_moving(shoulder, elbow, wrist):
    """
    Determines if the arm is at approximately 90° and actively moving.
    The elbow should be near 90° relative to the shoulder and wrist.
    """
    # Calculate vectors
    shoulder_to_elbow = np.array(elbow) - np.array(shoulder)
    elbow_to_wrist = np.array(wrist) - np.array(elbow)

    # Compute angle between shoulder-elbow and elbow-wrist vectors
    cosine_angle = np.dot(shoulder_to_elbow, elbow_to_wrist) / (
        np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist)
    )
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    # Consider arm movement if the angle is approximately 90 degrees ± 15 degrees
    return 75 <= angle <= 105

# Function to extract keypoints for the new video
def extract_keypoints(video_path):
    """
    Extract keypoints to analyze arm movement.
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

            # Extract relevant keypoints for left and right arms
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            # Check if arms are at 90° and actively moving
            left_arm_active = is_arm_at_90_and_moving(left_shoulder, left_elbow, left_wrist)
            right_arm_active = is_arm_at_90_and_moving(right_shoulder, right_elbow, right_wrist)

            # Store the binary values (0 or 1) for analysis
            keypoints.append([int(left_arm_active), int(right_arm_active)])

    cap.release()

    # Convert keypoints to padded sequence
    return pad_sequences([keypoints], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')

# Function to classify scores
def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.65:
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

        keypoints_padded = extract_keypoints(video_path)
        if keypoints_padded.size == 0:
            return {"video": video_path, "error": "No keypoints extracted"}

        # Reshape to match model input
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
    test_video_path = os.path.join("sprint_technique", "stages", "stage4", "test_videos", "1_user4.mp4")
    result = process_video(test_video_path)
    print(result)
