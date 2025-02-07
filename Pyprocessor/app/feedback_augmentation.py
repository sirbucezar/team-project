import os
import re
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
RESULTS_BLOB_CONTAINER = "results"

if not AZURE_STORAGE_CONNECTION_STRING:
    logger.error("AZURE_STORAGE_CONNECTION_STRING is not set. Exiting application.")
    raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is required.")

def extract_json_content(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0).strip())
            except json.JSONDecodeError:
                logger.error("Regex-extracted content is not valid JSON.")
        return None