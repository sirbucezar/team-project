import os
import sys
# Ensure that the parent directory (which contains the "app" package) is on sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force Transformers into offline mode so that no network look‐ups occur.
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import logging
import requests
import tempfile
import datetime
import torch

from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

# SocketIO imports (if you do not need them, you could remove these lines)
from flask_socketio import SocketIO
import eventlet
# Tell Eventlet not to monkey-patch DNS (this should help avoid the dnspython issue)
eventlet.monkey_patch(dns=False)

# Hugging Face & ML imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Azure imports
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient

# Additional imports for stage analysis
import cv2
import numpy as np
import mediapipe as mp

# Import stage scripts for shotput (adjust if you support other exercises)
from app.scripts.shotput_stage1_script import main as run_stage1
from app.scripts.shotput_stage2_script import main as run_stage2
from app.scripts.shotput_stage3_script import main as run_stage3
from app.scripts.shotput_stage4_script import main as run_stage4
from app.scripts.shotput_stage5_script import main as run_stage5

# Initialize Flask App
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# (Optional) Initialize SocketIO for real‑time logging.
socketio = SocketIO(app, cors_allowed_origins="*")
class SocketIOLoggingHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        socketio.emit("logs", {"message": log_entry})
socketio_handler = SocketIOLoggingHandler()
logger.addHandler(socketio_handler)

# Environment Variables and Constants
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# Ensure your local model folder is complete (with a valid config.json, weights, tokenizer files, etc.)
MODEL_PATH = "/app/llama7b-sports-coach-final.v2"
TABLE_NAME = "videoprocessing"
BLOB_CONTAINER_NAME = "results"

if not AZURE_STORAGE_CONNECTION_STRING:
    logger.error("Missing required environment variable: AZURE_STORAGE_CONNECTION_STRING")
    raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is required.")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model and Tokenizer from local files (offline mode)
try:
    logger.info("Loading tokenizer from local model folder...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    logger.info("Loading base model from local model folder...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,  # Use only local files.
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    logger.info("Merging LoRA adapter from local model folder...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH, local_files_only=True)
    model = model.merge_and_unload()
    model.to(device)

    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {str(e)}")
    raise

### Utility Functions ###

def update_processing_status(processing_id, status, message):
    """Update processing status in Azure Table Storage."""
    try:
        table_service = TableServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        table_client = table_service.get_table_client(TABLE_NAME)
        entity = {
            "PartitionKey": processing_id,
            "RowKey": "status",
            "Status": status,
            "Message": message,
            "Timestamp": datetime.datetime.utcnow().isoformat()
        }
        table_client.upsert_entity(entity)
        logger.info(f"Updated processing status for {processing_id} to '{status}'")
    except Exception as e:
        logger.error(f"Failed to update processing status: {str(e)}")

def upload_to_blob(processing_id, data):
    """Upload JSON results to Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        blob_name = f"{processing_id}/results.json"
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(json.dumps(data, indent=2), overwrite=True)
        logger.info(f"Uploaded results to blob: {blob_name}")
        return blob_name
    except Exception as e:
        logger.error(f"Failed to upload to blob: {str(e)}")
        return None

def download_video(url):
    """Download a remote video to a temporary file."""
    logger.info(f"Downloading video from {url}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
        logger.info(f"Downloaded video to {tmp_file.name}")
        return tmp_file.name
    except requests.RequestException as e:
        logger.error(f"Failed to download video: {str(e)}")
        raise

def cut_video_opencv(full_video_path: str, start_frame: int, end_frame: int) -> str:
    """
    Cut a subclip from the full video using OpenCV based on frame numbers.
    Returns the path to the temporary subclip file.
    """
    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {full_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    subclip_path = temp_file.name
    temp_file.close()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(subclip_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Reached end of video early at frame {frame_num}")
            break
        out.write(frame)
    cap.release()
    out.release()
    # Verify that the subclip can be opened
    subclip_cap = cv2.VideoCapture(subclip_path)
    if not subclip_cap.isOpened():
        subclip_cap.release()
        raise Exception("Could not open subclip video file")
    subclip_cap.release()
    return subclip_path

def analyze_stage(video_path: str, stage: dict) -> dict:
    """Analyze a single stage using the appropriate stage script."""
    stage_name = stage.get("name", "").lower()
    start_frame = int(stage.get("start_time", 0))
    end_frame = int(stage.get("end_time", 0))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    logger.info(f"Analyzing stage {stage_name} from frame {start_frame} to {end_frame} (fps: {fps})")
    subclip_path = cut_video_opencv(video_path, start_frame, end_frame)
    try:
        if stage_name == "stage1":
            return run_stage1(subclip_path)
        elif stage_name == "stage2":
            return run_stage2(subclip_path)
        elif stage_name == "stage3":
            return run_stage3(subclip_path)
        elif stage_name == "stage4":
            return run_stage4(subclip_path)
        elif stage_name == "stage5":
            return run_stage5(subclip_path)
        else:
            logger.error(f"Unknown stage: {stage_name}")
            return {"error": f"Unknown stage: {stage_name}"}
    finally:
        if os.path.exists(subclip_path):
            os.remove(subclip_path)

### Flask API Endpoints ###

@app.route("/analyze", methods=["POST"])
def analyze_video():
    """
    Handles video analysis requests.
    If the request JSON contains a "stages" key, performs stage‑by‑stage analysis;
    otherwise, returns a simple mock analysis result.
    """
    try:
        data = request.get_json(force=True)
        logger.info(f"Received /analyze request: {data}")
        processing_id = data.get("processing_id")
        video_url = data.get("video_url")
        if not processing_id or not video_url:
            logger.error("Missing processing_id or video_url.")
            return jsonify({"error": "processing_id and video_url are required."}), 400

        local_video_path = download_video(video_url)

        if "stages" in data:
            stages = data.get("stages", [])
            exercise = data.get("exercise", "shotput")
            result = {
                "processingId": processing_id,
                "exercise": exercise,
                "stageAnalysis": {},
                "metrics": {"overall_score": 0.0, "confidence": 0.0}
            }
            total_score = 0.0
            total_confidence = 0.0
            for stage in stages:
                stage_name = stage.get("name", "")
                stage_result = analyze_stage(local_video_path, stage)
                result["stageAnalysis"][stage_name] = stage_result
                score = stage_result.get("classified_score") or stage_result.get("score")
                confidence = stage_result.get("confidence", 0.0)
                if score is not None:
                    total_score += float(score)
                    total_confidence += float(confidence)
            result["metrics"]["overall_score"] = total_score
            result["metrics"]["confidence"] = total_confidence
        else:
            # Simple analysis branch (mocked result)
            result = {
                "processingId": processing_id,
                "stageAnalysis": {"example_stage": {"score": 1, "confidence": 0.95}},
                "metrics": {"overall_score": 1.0}
            }

        blob_url = upload_to_blob(processing_id, result)
        update_processing_status(processing_id, "completed", "Analysis completed")
        result["blob_url"] = blob_url
        return jsonify(result)
    except BadRequest as e:
        logger.error(f"Bad request: {str(e)}")
        return jsonify({"error": "Invalid JSON format."}), 400
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'local_video_path' in locals() and os.path.exists(local_video_path):
                os.remove(local_video_path)
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {str(cleanup_error)}")

@app.route("/augment", methods=["POST"])
def augment_feedback():
    """
    Handles augmentation requests using the fine‑tuned Llama‑2 model.
    """
    try:
        data = request.get_json(force=True)
        processing_id = data.get("processing_id", "unknown_id")
        feedback_data = data.get("stageAnalysis", {})
        user_prompt = data.get("user_prompt", "Provide actionable feedback.")
        if not feedback_data:
            return jsonify({"error": "Missing stageAnalysis data."}), 400

        # Prepare the prompt for the model
        model_prompt = f"""
Based on the provided data, generate structured feedback in JSON format.
Athletic Evaluation Data:
{json.dumps(feedback_data, indent=2)}
Ensure response strictly follows the JSON format below:
{{
    "stage": "<stage_name>",
    "criterion": "<criterion_description>",
    "score": "<0, 0.5, or 1>",
    "confidence": "<float between 0 and 1>",
    "feedback": {{
        "Observation": {{"title": "Observation", "body": "<detailed analysis>"}},
        "Improvement Suggestion": {{"title": "Improvement Suggestion", "body": "<corrections>"}},
        "Justification": {{"title": "Justification", "body": "<scientific reasoning>"}},
        "Encouragement": {{"title": "Encouragement", "body": "<motivation>"}}
    }},
    "injury_risk": {{"high_risk": <true/false>, "disclaimer": "<injury concerns>"}},
    "visualization_tip": "<mental imagery>"
}}
"""
        inputs = tokenizer(model_prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=1024,
            num_beams=3,
            temperature=0.7,
            early_stopping=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        feedback = json.loads(generated_text)
        blob_url = upload_to_blob(processing_id, feedback)
        return jsonify({
            "status": "Feedback generated successfully",
            "processing_id": processing_id,
            "feedback": feedback,
            "blob_url": blob_url
        })
    except BadRequest:
        return jsonify({"error": "Invalid JSON format."}), 400
    except Exception as e:
        logger.error(f"Error in /augment endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": os.environ.get("VERSION", "dev")
    }), 200

if __name__ == "__main__":
    # For local testing with SocketIO support; if not needed, replace with app.run(...)
    socketio.run(app, host="0.0.0.0", port=80)