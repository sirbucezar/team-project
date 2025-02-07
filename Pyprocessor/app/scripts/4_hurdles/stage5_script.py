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
MODEL_PATH = os.path.join("hurdling", "stages", "stage5", "models", "hurdles_stage5.keras")
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

# Function to check if a large stride is achieved after hurdle clearance
def is_large_stride(hip, knee, ankle):
    """
    Determines if a large stride is achieved by analyzing the relative distance 
    between the hip, knee, and ankle positions.
    """
    hip_x, hip_y = hip
    knee_x, knee_y = knee
    ankle_x, ankle_y = ankle

    # Large stride condition: ankle should be significantly ahead of hip and knee
    stride_length = ankle_x - hip_x > 0.3 and ankle_x - knee_x > 0.2
    return stride_length

# Function to extract keypoints for large stride analysis
def extract_keypoints(video_path):
    """
    Extract keypoints for large stride analysis.
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

            # Extract keypoints for large stride analysis
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            # Check if a large stride is achieved
            left_stride = is_large_stride(left_hip, left_knee, left_ankle)
            right_stride = is_large_stride(right_hip, right_knee, right_ankle)

            # Store the binary values (0 or 1) for analysis
            keypoints.append([int(left_stride), int(right_stride)])

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
    test_video_path = os.path.join("hurdling", "stages", "stage5", "test_videos", "0_user21.mp4")
    result = process_video(test_video_path)
    print(result)
