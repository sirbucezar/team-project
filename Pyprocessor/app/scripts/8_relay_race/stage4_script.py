import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_PATH = os.path.join("relay_race", "stage4_models", "stage4-final.h5")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
else:
    print(f"Model file exists at {MODEL_PATH}")

# Custom LSTM layer to handle time_major argument
def custom_LSTM(**kwargs):
    if 'time_major' in kwargs:
        kwargs.pop('time_major')  # Remove time_major argument
    return LSTM(**kwargs)

# Load the model with the custom LSTM
model = load_model(MODEL_PATH, custom_objects={'LSTM': custom_LSTM})

MAX_SEQUENCE_LENGTH = model.input_shape[1]
NUM_FEATURES = model.input_shape[-1]  # Number of features

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculate the angle formed by three points a, b, c (b is the vertex)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Unable to open video file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    keypoints_list = []

    frame_count = 0
    keypoints_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract relevant keypoints (including wrists)
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x * frame_width, landmarks[mp_pose.PoseLandmark.NOSE].y * frame_height]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame_width, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame_width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height]

            # Calculate angles and movements
            head_angle = calculate_angle(left_shoulder, nose, right_shoulder)
            hip_movement = right_hip[0] - left_hip[0]  # Track x movement
            wrist_distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))  # Distance between wrists

            # Append 5 features per frame (remove one feature)
            keypoints_list.append([
                head_angle,  # 1. Head angle
                hip_movement,  # 2. Hip movement (horizontal)
                wrist_distance,  # 3. Wrist distance (could represent arm movement)
                nose[0] / frame_width,  # 4. Normalized nose x
                nose[1] / frame_height,  # 5. Normalized nose y
            ])

            # Additional information for debugging/output
            is_exchange = 1 if frame_count % 50 == 0 else 0  # For demonstration, assuming exchange every 50th frame
            processed_frame_count = frame_count

            keypoints_data.append({
                "frame": processed_frame_count,
                "is_exchange": is_exchange,
                "head_angle": head_angle,
                "hip_movement": hip_movement,
                "wrist_distance": wrist_distance,
                "nose_x": nose[0],
                "nose_y": nose[1],
                "left_shoulder_x": left_shoulder[0],
                "left_shoulder_y": left_shoulder[1],
                "right_shoulder_x": right_shoulder[0],
                "right_shoulder_y": right_shoulder[1],
                "left_hip_x": left_hip[0],
                "left_hip_y": left_hip[1],
                "right_hip_x": right_hip[0],
                "right_hip_y": right_hip[1],
                "left_wrist_x": left_wrist[0],
                "left_wrist_y": left_wrist[1],
                "right_wrist_x": right_wrist[0],
                "right_wrist_y": right_wrist[1],
            })

    cap.release()

    return keypoints_list, keypoints_data


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

        # Extract keypoints
        keypoints_list, keypoints_data = extract_keypoints(video_path)

        if not keypoints_list:
            return {"video": video_path, "error": "No keypoints extracted"}

        # Prepare keypoints for classification
        keypoints_padded = pad_sequences([keypoints_list], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')

        # Reshape after padding to match the input shape of the model
        keypoints_padded = keypoints_padded.reshape((1, MAX_SEQUENCE_LENGTH, NUM_FEATURES))

        # Make predictions
        predictions = model.predict(keypoints_padded)
        predicted_score = float(predictions[0][0])
        classified_score = classify_score(predicted_score)

        return {
            "video": video_path,
            "predicted_score": predicted_score,
            "classified_score": classified_score
        }
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {"video": video_path, "error": str(e)}

if __name__ == "__main__":
    test_video_path = os.path.join("/Users/danyukezz/Desktop/models_copy/relay_race/stage4/user8.mp4")
    result = process_video(test_video_path)
    print(result)
