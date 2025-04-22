import json
import requests
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2 # OpenCV for drawing
import matplotlib.pyplot as plt
import io
import os
import argparse # Added for command-line arguments

def visualize_sam_output(json_path, image_url_or_path, show_bboxes=False):
    """
    Loads segmentation results from a JSON file and visualizes the masks
    (and optionally bounding boxes) on the corresponding image.

    Args:
        json_path (str): Path to the segmentation_result.json file.
        image_url_or_path (str): The URL or local file path of the original image.
        show_bboxes (bool): If True, draw bounding boxes around the masks.
    """
    # --- 1. Load the Segmentation Data ---
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        masks_data = data.get("masks")
        if masks_data is None:
            print(f"Error: 'masks' key not found in {json_path}")
            return
        if not isinstance(masks_data, list):
             print(f"Error: 'masks' key in {json_path} is not a list.")
             return
        print(f"Loaded {len(masks_data)} mask entries from {json_path}")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading JSON: {e}")
        return

    # --- 2. Load the Original Image ---
    try:
        if image_url_or_path.startswith(('http://', 'https://')):
            print(f"Fetching image from URL: {image_url_or_path}")
            response = requests.get(image_url_or_path, stream=True, timeout=10)
            response.raise_for_status()
            img_pil = Image.open(io.BytesIO(response.content))
        else:
            print(f"Loading image from local path: {image_url_or_path}")
            if not os.path.exists(image_url_or_path):
                 print(f"Error: Image file not found at {image_url_or_path}")
                 return
            img_pil = Image.open(image_url_or_path)

        image_np = np.array(img_pil.convert('RGB'))
        print(f"Image loaded successfully. Shape: {image_np.shape}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image URL {image_url_or_path}: {e}")
        return
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_url_or_path}")
        return
    except UnidentifiedImageError:
        print(f"Error: Could not identify or open image file/data from {image_url_or_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading the image: {e}")
        return

    # --- 3. Draw Masks and BBoxes on the Image ---
    overlay = image_np.copy()
    output = image_np.copy() # Draw bboxes directly onto this later if needed
    alpha = 0.5
    num_masks = len(masks_data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_masks))[:, :3]
    bbox_color_bgr = (0, 255, 0) # Green for BBoxes (BGR format for OpenCV)
    bbox_thickness = 2

    mask_count = 0
    bbox_count = 0
    for i, mask_info in enumerate(masks_data):
        polygon_points = mask_info.get("polygon")
        bbox_data = mask_info.get("bounding_boxes") # Expecting [xmin, ymin, width, height]

        # Draw Polygon (Mask)
        if polygon_points and isinstance(polygon_points, list) and len(polygon_points) >= 3:
            try:
                pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
                color_rgb_0_1 = colors[i]
                color_bgr_0_255 = tuple(int(c * 255) for c in color_rgb_0_1[::-1])
                cv2.fillPoly(overlay, [pts], color_bgr_0_255)
                mask_count += 1
            except ValueError as e:
                 print(f"Warning: Skipping mask {i} polygon due to error converting points: {e}")
                 continue
            except Exception as e:
                 print(f"Warning: An unexpected error occurred drawing mask {i} polygon: {e}")
                 continue
        else:
            print(f"Warning: Skipping mask {i} polygon due to missing or invalid 'polygon' data.")

        # Draw Bounding Box (if requested and available)
        if show_bboxes:
            if bbox_data and isinstance(bbox_data, list) and len(bbox_data) == 4:
                try:
                    x_min, y_min, width, height = map(int, bbox_data)
                    x_max = x_min + width
                    y_max = y_min + height
                    # Draw rectangle directly on the 'output' image (which will be blended later)
                    cv2.rectangle(output, (x_min, y_min), (x_max, y_max), bbox_color_bgr, bbox_thickness)
                    bbox_count += 1
                except (ValueError, TypeError) as e:
                    print(f"Warning: Skipping bbox for mask {i} due to invalid data format: {e}. Data: {bbox_data}")
                except Exception as e:
                     print(f"Warning: An unexpected error occurred drawing bbox {i}: {e}")
            else:
                print(f"Warning: Skipping bbox for mask {i} due to missing or invalid 'bounding_boxes' data. Found: {bbox_data}")

    # Blend the overlay with the original image (now potentially containing bboxes)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    print(f"Drew {mask_count} valid masks" + (f" and {bbox_count} valid bounding boxes." if show_bboxes else " onto the image."))

    # --- 4. Display the Result ---
    plt.figure(figsize=(12, 10))
    plt.imshow(output)
    title = f"Segmentation Masks from {os.path.basename(json_path)} on {os.path.basename(image_url_or_path)}"
    if show_bboxes:
        title += " (with Bounding Boxes)"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Example Usage --- Adjust visualize_segmentation call below this line ---
if __name__ == "__main__":
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description="Visualize segmentation masks (and optionally bounding boxes) from a JSON file on an image.")
    parser.add_argument("json_path", help="Path to the segmentation_result.json file.")
    parser.add_argument("image_location", help="URL or local file path of the original image.")
    parser.add_argument("--show-bboxes", action="store_true", help="Display bounding boxes in addition to masks.")

    args = parser.parse_args()

    # --- Run Visualization --- Pass the command line arg to the function
    visualize_sam_output(args.json_path, args.image_location, args.show_bboxes) 