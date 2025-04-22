import requests
import io
import os
import hashlib
from flask import request, Blueprint
from util import build_response_error, generate_polygons_from_masks, select_device
from cds import fetch_from_cds, save_to_cds
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
# # sam2_checkpoint = "app/app/checkpoint/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

segment_routes = Blueprint("segment_routes", __name__)

# Define cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segment_cache")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)


def generate_cache_file_path(url):
    """Generate a unique file for the cache based on the image URL."""
    hashed_url = hashlib.md5(url.encode("utf-8")).hexdigest()
    return hashed_url


@segment_routes.route("/generate_masks", methods=["POST"])
def generate_masks():
    """
    Main route that uses SAM version 2 to generate masks for a single image.

    Route: /api/v1/generate_masks

    @todo Remove print to use a logger class.

    @todo add support to CDS images.

    @todo change cache approach.
    """
    environment = os.getenv("ENV", "prod")
    device = select_device()

    print("SEGMENT_SELECTED_DEVICE:", device)

    input = request.get_json()
    image_url = input["image"]
    storage_token = input["storageRefsToken"]

    cache_file_path = generate_cache_file_path(image_url)

    try:
        response = requests.get(image_url)

        print(f"Image URL Response Code: {response.status_code}")

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_width, image_height = image.size
        image_np = np.array(image)

        print("Start Mask Generator")
        print("Cache File Path", cache_file_path)

        has_cached_data = fetch_from_cds(cache_file_path, storage_token)
        if has_cached_data is not None:
            sam_result = has_cached_data
        else:
            sam2 = build_sam2(
                model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
            )

            if environment == "dev":
                mask_generator = SAM2AutomaticMaskGenerator(sam2)
                print("Initialized 'dev' Mask Generator")
            else:
                # This configuration generates better masks
                # However, without using cuda there is a performance issue
                # Since not all devices support cuda
                # Only prod need to use this configuration
                mask_generator = SAM2AutomaticMaskGenerator(
                    model=sam2,
                    points_per_side=64,
                    points_per_batch=128,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.92,
                    stability_score_offset=0.7,
                    crop_n_layers=1,
                    box_nms_thresh=0.7,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=25.0,
                    use_m2m=True,
                )
                print("Initialized 'prod' Mask Generator")
            print("Generating masks..")
            sam_result = mask_generator.generate(image_np)
            print("Generated masks!")
            save_to_cds(cache_file_path, sam_result, storage_token)

        print("Mask Generator Finished")

        mask_data_list = []
        for idx, mask_data in enumerate(sam_result):
            mask_info = {
                "id": idx,
                "bounding_boxes": mask_data["bbox"],
                "point_coords": mask_data["point_coords"],
                "segmentation": mask_data["segmentation"],
            }
            mask_data_list.append(mask_info)

        # print("Generate polygons from masks", len(mask_data_list))
        print("Generating polygons from masks...")
        valid_polygons = generate_polygons_from_masks(
            mask_data_list, image_width, image_height
        )
        print("Generated polygons from masks!!")

        final_response = {"masks": valid_polygons}

        return final_response

    except Exception as e:
        message = f"Failed to predict segmentation: {e}"
        return build_response_error(message)