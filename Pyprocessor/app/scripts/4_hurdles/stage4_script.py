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
MODEL_PATH = os.path.join("hurdling", "stages", "stage4", "models", "hurdles_stage4.keras")
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

# Function to check if torso and opposite arm align with the lead leg
def is_torso_arm_aligned(hip, shoulder, arm):
    """
    Determines if the torso and opposite arm align with the lead leg by checking
    if the shoulder, hip, and arm are in a straight line or close alignment.
    """
    hip_x, hip_y = hip
    shoulder_x, shoulder_y = shoulder
    arm_x, arm_y = arm

    # Alignment condition: Shoulder, hip, and arm should be nearly aligned vertically
    return abs(hip_x - shoulder_x) < 0.05 and abs(arm_x - shoulder_x) < 0.05

# Function to extract keypoints for alignment and extension analysis
def extract_keypoints(video_path):
    """
    Extract keypoints for torso-arm alignment and lead leg extension.
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

            # Extract keypoints for torso, arms, and legs
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            right_arm = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            left_arm = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            # Check alignment between torso, arm, and lead leg
            left_alignment = is_torso_arm_aligned(left_hip, left_shoulder, right_arm)
            right_alignment = is_torso_arm_aligned(right_hip, right_shoulder, left_arm)

            # Store the binary values (0 or 1) for analysis
            keypoints.append([int(left_alignment), int(right_alignment)])

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
    test_video_path = os.path.join("hurdling", "stages", "stage4", "test_videos", "1_user5.mp4")
    result = process_video(test_video_path)
    print(result)
