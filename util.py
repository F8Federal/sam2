import cv2
import base64
import torch
from flask import jsonify
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
from io import BytesIO


def build_response_error(message, status=422):
    response = jsonify({"error": message})
    response.status_code = status
    return response


def mask_to_base64(mask, image_width, image_height):
    """
    Convert a binary mask to a base64-encoded PNG image.

    Parameters:
    - mask (np.ndarray): A binary (0 or 1) numpy array representing the mask.
    - image_width (int): The width of the target image.
    - image_height (int): The height of the target image.

    Returns:
    - str: A base64-encoded string representing the PNG image.
    """
    if mask.shape[0] != image_height or mask.shape[1] != image_width:
        mask = np.resize(mask, (image_height, image_width))

    mask_image = Image.fromarray(np.uint8(mask * 255), mode="L")

    buffer = BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_polygons_from_masks(masks_data, image_width, image_height):
    """
    Generate polygons from binary masks and encode the masks as base64 PNGs.

    Parameters:
    - masks_data (List[Dict[str, Any]]): A list of dictionaries, each containing:
        - 'segmentation' (np.ndarray): A binary mask (0s and 1s) for segmentation.
        - 'point_coords' (List[List[int]]): Coordinates of key points in the mask.
        - 'bounding_boxes' (List[List[int]]): Bounding box coordinates for the mask.
    - image_width (int): The width of the image.
    - image_height (int): The height of the image.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries containing:

        - 'id' (int): The index of the mask in the input list.
        - 'polygon' (List[Tuple[int, int]]): The exterior coordinates of the polygon.
        - 'point_coords' (List[List[int]]): The original point coordinates.
        - 'bounding_boxes' (List[List[int]]): The bounding box coordinates.
        - 'mask_base64' (str): The base64-encoded PNG representation of the mask.
    """
    valid_polygons = []

    for idx, mask_data in enumerate(masks_data):
        mask = mask_data["segmentation"]
        mask_binary = np.uint8(mask * 255)
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(simplified_contour) >= 3:
                contour_points = simplified_contour.squeeze().tolist()
                mask_polygon = Polygon(contour_points)

                if mask_polygon.is_valid:
                    mask_base64 = mask_to_base64(mask, image_width, image_height)

                    valid_polygons.append(
                        {
                            "id": idx,
                            "polygon": list(mask_polygon.exterior.coords),
                            "point_coords": mask_data.get("point_coords", []),
                            "bounding_boxes": mask_data.get("bounding_boxes", []),
                            "mask_base64": mask_base64,
                        }
                    )

    return valid_polygons


def select_device():
    """
    Select device for SAM 2 model, available options: cuda, mps, cpu.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    return device