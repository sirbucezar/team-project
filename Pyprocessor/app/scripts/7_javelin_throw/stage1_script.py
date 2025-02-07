import logging
import os
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join("spear_throwing", "stages", "stage1", "models", "javelin_stage1.keras")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

model = load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_mse': lambda y_true, y_pred: K.mean(K.switch(y_true < 0.70, 2.0, 1.0) * K.square(y_true - y_pred))
    },
)
MAX_SEQUENCE_LENGTH = model.input_shape[1]
NUM_FEATURES = model.input_shape[-1]  # Number of features

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to determine if the javelin is brought backward
def is_javelin_brought_backward(shoulder, wrist):
    """
    Determines if the javelin (wrist) is positioned behind the shoulder in the x-axis,
    indicating that the javelin is brought backward.
    """
    shoulder_x, _ = shoulder
    wrist_x, _ = wrist
    return wrist_x < shoulder_x

# Function to extract keypoints for javelin backward analysis
def extract_keypoints(video_path):
    """
    Extract keypoints to analyze if the javelin is brought backward.
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

            # Extract keypoints for the right shoulder and right wrist
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            # Check if the javelin is brought backward
            javelin_backward = is_javelin_brought_backward(right_shoulder, right_wrist)

            # Store the binary value (0 or 1) for analysis
            keypoints.append([int(javelin_backward)])

    cap.release()

    # Convert keypoints to padded sequence
    return pad_sequences([keypoints], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')

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
    test_video_path = os.path.join("spear_throwing", "stages", "stage1", "test_videos", "0_user6.mp4")
    result = process_video(test_video_path)
    print(result)
