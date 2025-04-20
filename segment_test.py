import requests
import json

# URL of your Flask server
server_url = 'http://localhost:5000/api/v1/generate_masks'

# Use a publicly accessible image for testing
test_image_url = '/Users/jamesemilian/Desktop/cronus-ai/figure8/anomaly_detect/data/coco_data_prep/coco_val_1pct/data/000000000321.jpg'

# Request payload with dummy token
payload = {
    'image': test_image_url,
    'storageRefsToken': 'test-token'  # Replace with actual token if available
}

# Send the request
response = requests.post(server_url, json=payload)

# Print the response
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    # Save the response to a file for inspection since it might be large
    with open('./segmentation_response.json', 'w') as f:
        json.dump(response.json(), indent=2, f)
    print("Response saved to segmentation_response.json")
else:
    print(response.text)