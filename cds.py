import requests
import os
import pickle
import json
import gzip
from dotenv import load_dotenv

load_dotenv()


def fetch_from_cds(hashed_url, storageToken, extension="pkl"):
    # """
    # Fetch cached mask data from Customer Data Service.

    # Args:
    #     hashed_url (str): The hashed URL to use as the file identifier

    # Returns:
    #     dict or None: The loaded pickle data if successful, None otherwise
    # """
    # requestor_proxy_url = os.getenv("REQUESTOR_PROXY_URL")

    # params = f"storageToken={storageToken}&path={hashed_url}.{extension}"
    # url = f"{requestor_proxy_url}/v1/annotations/validate_model_cache?{params}"

    # try:
    #     response = requests.get(url)
    #     content = json.loads(response.content)
    #     saved = requests.get(content["upload_url"])

    #     if saved.status_code == 200:
    #         decompressed_data = gzip.decompress(saved.content)
    #         if extension == "pkl":
    #             return pickle.loads(decompressed_data)
    #         else:
    #             return json.load(decompressed_data)
    #     else:
    #         print(f"CDS request failed with status code: {saved.status_code}")
    #         return None
    # except Exception as e:
    #     print(f"Error fetching from CDS: {str(e)}")
    #     return None
    """Mock implementation that always returns None to force processing"""
    return None


def save_to_cds(hashed_url, data, token, extension="pkl"):
    # """
    # Save mask data to Customer Data Service.
    # This function would implement the PUT/POST request to save data to CDS.

    # Args:
    #     hashed_url (str): The hashed URL to use as the file identifier
    #     data: The data to save

    # Returns:
    #     bool: True if successful, False otherwise
    # """
    # requestor_proxy_url = os.getenv("REQUESTOR_PROXY_URL")

    # url = f"{requestor_proxy_url}/v1/annotations/cache_model_upload?storageToken={token}&path={hashed_url}.{extension}"

    # if extension == "pkl":
    #     pickled_data = pickle.dumps(data)
    #     compressed_data = gzip.compress(pickled_data)
    # else:
    #     pickled_data = json.dump(data)
    #     compressed_data = gzip.compress(pickled_data)

    # try:
    #     response = requests.get(url)
    #     content = json.loads(response.content)

    #     requests.put(content["upload_url"], data=compressed_data)

    #     return True
    # except Exception as e:
    #     print(f"Error fetching from CDS: {str(e)}")
    #     return None
    """Mock implementation that just returns success"""
    return True
