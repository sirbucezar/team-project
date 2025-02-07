import os
import re
import json
import logging
import requests
import tempfile
from app.storage_utils import safe_upload_blob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", "cropped-videos")
RESULTS_CONTAINER_NAME = os.getenv("RESULTS_CONTAINER_NAME", "results")

if not AZURE_STORAGE_CONNECTION_STRING:
    logger.error("AZURE_STORAGE_CONNECTION_STRING is not set. Exiting application.")
    raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is required.")

def get_local_path(video_url: str) -> str:
    if video_url.startswith("http://") or video_url.startswith("https://"):
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                return tmp_file.name
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise
    elif os.path.exists(video_url):
        return video_url
    else:
        raise FileNotFoundError(f"Local video file not found: {video_url}")