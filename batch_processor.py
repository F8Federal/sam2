import os
import cv2
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import logging
from tqdm import tqdm
import json

# Assuming sam_scripts is in the same parent directory or PYTHONPATH includes it
from sam2.util import generate_polygons_from_masks

# Fallback if running from a different structure, adjust as necessary
# print("Warning: Could not import from sam2_scripts. Ensure it's in the Python path.")
# Define minimal fallbacks or raise error if essential
def select_device(device_str='auto'):
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    # Add MPS check if needed
    elif device_str == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# def generate_polygons_from_masks(masks, image_shape):
#     # Minimal fallback - requires proper implementation or ensures sam_scripts is available
#     print("Warning: generate_polygons_from_masks not imported. Polygon generation will be skipped.")
#     # Returning raw masks for now, but ideally, this dependency should be resolved
#     return masks 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def generate_masks_for_directory(
    image_dir: str,
    model_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml", # Default model config
    model_checkpoint: str = "checkpoints/sam2.1_hiera_tiny.pt", # Default checkpoint
    device_str: str = "auto",
    mask_gen_kwargs: dict | None = None, # Allow customizing mask generator params
) -> dict:  
    """
    Generates masks for all supported images in a directory using SAM2AutomaticMaskGenerator.

    Args:
        image_dir: Path to the directory containing images.
        model_config: Path or identifier for the SAM 2 model configuration yaml.
        model_checkpoint: Path to the SAM 2 model checkpoint (.pt) file.
        device_str: Device to run inference on ('auto', 'cuda', 'mps', 'cpu').
        mask_gen_kwargs: Optional dictionary of keyword arguments to pass to
                         Sam2AutomaticMaskGenerator constructor.

    Returns:
        A dictionary where keys are image filenames (basename) and values are
        lists of processed mask dictionaries (containing polygons, etc.).
    """
    if not os.path.isdir(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        return {}

    # Handle mask generator arguments - use defaults similar to segment.py 'dev'
    if mask_gen_kwargs is None:
         mask_gen_kwargs = {
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 100, # Default value from SAM
            "output_mode": "binary_mask" # Ensure we get masks for polygon conversion
        }

    device = select_device(device_str=device_str)
    logging.info(f"Using device: {device}")

    try:
        logging.info("Loading SAM 2 model...")
        sam2 = build_sam2(model_config, model_checkpoint, device=device, apply_postprocessing=False)
        # mask_generator = Sam2AutomaticMaskGenerator(model, **mask_gen_kwargs)
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load SAM 2 model: {e}")
        return {}

    all_masks_data = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not image_files:
        logging.warning(f"No supported image files found in {image_dir}")
        return {}

    logging.info(f"Found {len(image_files)} images to process.")

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, filename)
        image_name = os.path.basename(filename) # Use basename as key
        logging.info(f"Processing {image_name}...")

        try:
            # Load image using OpenCV (consistent with segment.py)
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Could not read image: {image_path}. Skipping.")
                continue
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_shape = image_rgb.shape[:2] # (height, width)

            logging.info(f"Generating masks for {image_name}...")
            # Generate raw masks
            raw_masks = mask_generator.generate(image_rgb)
            logging.info(f"Generated {len(raw_masks)} raw masks for {image_name}.")

            if not raw_masks:
                logging.warning(f"No masks generated for {image_name}. Skipping polygon conversion.")
                all_masks_data[image_name] = []
                continue

            # Process masks to get polygons, base64 etc.
            logging.info(f"Converting masks to polygons for {image_name}...")
            processed_masks = generate_polygons_from_masks(raw_masks, image_shape)
            logging.info(f"Successfully processed masks for {image_name}.")

            all_masks_data[image_name] = processed_masks

        except ImportError as e:
             logging.error(f"ImportError during processing {image_name}: {e}. Make sure sam_scripts is accessible.")
             # Handle case where util functions weren't imported
             all_masks_data[image_name] = [] # Or store raw masks if preferred
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}", exc_info=True)
            all_masks_data[image_name] = [] # Store empty list on error

    logging.info("Finished processing all images.")
    return all_masks_data

def convert_masks_to_labels_format(mask_data: dict, image_dir: str | None = None) -> list:
    """
    Converts the mask data dictionary (output of generate_masks_for_directory)
    into a list format similar to labels.json, with empty 'label' fields.

    Args:
        mask_data: Dictionary output from generate_masks_for_directory.
                     Keys are image filenames, values are lists of processed masks.
        image_dir: Optional. If provided, the 'image_path' in the output will be relative
                   to this directory (e.g., 'images/cat.jpg'). Otherwise, it will
                   just be the filename (e.g., 'cat.jpg').

    Returns:
        A list of dictionaries, where each dictionary represents an image and its
        shapes (masks) in the label-studio compatible format, ready for labeling.
    """
    labels_output = []
    logging.info(f"Converting {len(mask_data)} images' mask data to labels format...")

    for image_filename, masks_list in mask_data.items():
        if not masks_list:
            logging.warning(f"Skipping {image_filename} in labels conversion as it has no masks.")
            continue

        # Construct the image path for the output JSON
        if image_dir:
            # Create a relative path from the provided base directory
            relative_image_path = os.path.join(os.path.basename(os.path.normpath(image_dir)), image_filename)
        else:
            # Just use the filename if no base directory is given
            relative_image_path = image_filename

        image_entry = {
            "image_path": relative_image_path.replace("\\", "/"), # Ensure forward slashes
            "shapes": []
        }

        for mask in masks_list:
            # Ensure polygon data exists and is not empty
            if "polygon" not in mask or not mask["polygon"]:
                logging.warning(f"Mask ID {mask.get('id', 'N/A')} for {image_filename} has no polygon data. Skipping.")
                continue

            shape_entry = {
                "label": "", # Intentionally empty for later prediction/labeling
                "points": mask["polygon"],
                "shape_type": "polygon",
                "flags": { # Optional: Add more info here if needed later
                    "sam_id": mask.get("id"),
                    # "bounding_box": mask.get("bounding_boxes") # Uncomment if needed
                },
                "group_id": None # Standard field in label-studio format
            }
            image_entry["shapes"].append(shape_entry)

        if image_entry["shapes"]: # Only add image entry if it has valid shapes
            labels_output.append(image_entry)
        else:
             logging.warning(f"No valid shapes converted for {image_filename}. Excluding from labels output.")


    logging.info("Finished converting mask data to labels format.")
    return labels_output


# Example Usage (Optional - can be commented out or placed under if __name__ == '__main__':)
if __name__ == '__main__':
    # Adjust paths as needed
    IMAGE_DIRECTORY = "/Users/jamesemilian/Desktop/cronus-ai/figure8/anomaly_detect/data/coco_data_prep/coco_val_1pct/data" # Assume images are in ./images/
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml" # Or path to your config
    MODEL_CKPT = "checkpoints/sam2.1_hiera_tiny.pt" # Or path to your checkpoint
    OUTPUT_MASKS_RAW_JSON = "../sam2_outputs/generated_masks_raw.json"
    OUTPUT_LABELS_JSON = "../sam2_outputs/generated_labels_empty.json"

    if not os.path.exists(IMAGE_DIRECTORY):
        print(f"Error: Image directory '{IMAGE_DIRECTORY}' not found.")
    elif not os.path.exists(MODEL_CKPT):
        print(f"Error: Model checkpoint '{MODEL_CKPT}' not found.")
    else:
        # 1. Generate masks for all images in the directory
        mask_results = generate_masks_for_directory(
            image_dir=IMAGE_DIRECTORY,
            model_config=MODEL_CFG,
            model_checkpoint=MODEL_CKPT
            # Add other args like device_str='cuda' if needed
        )

        if mask_results:
            # Optional: Save the raw mask dictionary (can be large)
            try:
                with open(OUTPUT_MASKS_RAW_JSON, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization if any remain
                    # (generate_polygons_from_masks should handle this, but as a safeguard)
                    serializable_results = json.loads(json.dumps(mask_results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
                    json.dump(serializable_results, f, indent=2) # Use indent=2 for potentially smaller file size
                print(f"Raw mask generation complete. Results saved to {OUTPUT_MASKS_RAW_JSON}")
            except TypeError as e:
                print(f"Error saving raw mask results to JSON: {e}. Data might contain non-serializable types.")
            except Exception as e:
                 print(f"An unexpected error occurred during raw mask saving: {e}")

            # 2. Convert the mask results to the labels.json format
            labels_formatted_data = convert_masks_to_labels_format(mask_results, image_dir=IMAGE_DIRECTORY)

            # 3. Save the labels format JSON
            if labels_formatted_data:
                try:
                    with open(OUTPUT_LABELS_JSON, 'w') as f:
                        json.dump(labels_formatted_data, f, indent=2)
                    print(f"Conversion to labels format complete. File saved to {OUTPUT_LABELS_JSON}")
                except Exception as e:
                    print(f"An unexpected error occurred during labels format saving: {e}")
            else:
                 print("Conversion to labels format resulted in empty data. No file saved.")

        else:
            print("Mask generation finished, but no results were produced.") 