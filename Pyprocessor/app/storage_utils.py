import os
import json
import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
from tenacity import retry, wait_exponential, stop_after_attempt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
RESULTS_BLOB_CONTAINER = "results"

if not AZURE_STORAGE_CONNECTION_STRING:
    raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is required.")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def safe_upload_blob(blob_client, data, overwrite=True):
    blob_client.upload_blob(data, overwrite=overwrite)

def upload_to_blob(processing_id, data):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(RESULTS_BLOB_CONTAINER)
        blob_name = f"{processing_id}/results.json"
        blob_client = container_client.get_blob_client(blob_name)
        results_json = json.dumps(data, indent=2)
        safe_upload_blob(blob_client, results_json, overwrite=True)
        logger.info(f"Uploaded results to blob: {blob_name}")
        return blob_name
    except AzureError as e:
        logger.error(f"Failed to upload blob: {str(e)}")
        return None