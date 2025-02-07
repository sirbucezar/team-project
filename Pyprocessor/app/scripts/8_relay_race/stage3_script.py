import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from keras.layers import LSTM
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_PATH = os.path.join("relay_race", "stage2_models", "stage2-final.h5")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
else:
    print(f"Model file exists at {MODEL_PATH}")

# Custom LSTM layer to handle time_major argument
def custom_LSTM(**kwargs):
    if 'time_major' in kwargs:
        kwargs.pop('time_major')
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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x * frame_width, landmarks[mp_pose.PoseLandmark.NOSE].y * frame_height]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height]

            # Calculate features
            head_angle = calculate_angle(left_shoulder, nose, right_shoulder)
            wrist_distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
            
            # Remove the flag feature (last entry in the original list)
            keypoints_list.append([
                head_angle,  # 1. Head angle
                wrist_distance,  # 2. Wrist distance
                nose[0] / frame_width,  # 3. Normalized nose x
                nose[1] / frame_height,  # 4. Normalized nose y
                # Removed the flag feature to reduce it to 12 features
            ])

    cap.release()
    return keypoints_list

def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.7:
        return 1.0
    elif prediction >= 0.35:
        return 0.5
    else:
        return 0.0
    
def process_video(video_path):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Error: Video file not found at {video_path}")

        keypoints_list = extract_keypoints(video_path)
        if not keypoints_list:
            return {"video": video_path, "error": "No keypoints extracted"}

        # Dynamically adjust the number of features per frame
        num_features = len(keypoints_list[0])  # Get number of features from the first frame
        print(f"Number of frames: {len(keypoints_list)}")
        print(f"Features in a frame: {num_features}")

        # Pad or repeat the features to 13 dynamically
        target_features = 13
        padded_keypoints_list = [k + [0] * (target_features - len(k)) if len(k) < target_features else k[:target_features] for k in keypoints_list]

        # Pad or truncate the sequence length to match MAX_SEQUENCE_LENGTH
        keypoints_padded = pad_sequences([padded_keypoints_list], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')

        # Debug shape before reshaping
        print(f"Shape before reshaping: {keypoints_padded.shape}")

        # Reshape to match the model's expected input
        keypoints_padded = keypoints_padded.reshape((1, MAX_SEQUENCE_LENGTH, target_features))  # 13 features per frame

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
    test_video_path = os.path.join("relay_race", "stage1-test", "user1_output.mp4")
    result = process_video(test_video_path)
    print(result)
