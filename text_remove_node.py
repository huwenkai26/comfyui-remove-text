import os

import cv2
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image

from .dbnet.dbnet_infer import textRemove

class ImageRemoveTextNode:
    """
    A simple node that retrieves basic information from an input image
    """

    # Define node's basic information
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Output a string
    FUNCTION = "remove_text"  # Processing function name

    CATEGORY = "image"  # Node category
    TITLE = "Image Info"  # Node title
    DESCRIPTION = "Get basic information from the input image"  # Node description

    def remove_text(self, image):
        # Convert tensor to PIL Image
        print("remove text node")
        pil_image = image.cpu().numpy().astype(np.float32)
        print("remove pil_image")
        # Get image information
        if len(image) > 1:
            raise Exception(f'Only images with batch_size==1 are supported! batch_size={len(image)}')
        nparray = (image[0].cpu().numpy()[..., ::-1] * 255).astype(np.uint8)  # reverse color channel from RGB to BGR
        # Convert tensor to PIL Image

        # Convert PIL Image to NumPy array for OpenCV processing
        img = cv2.cvtColor(nparray, cv2.COLOR_RGB2BGR)
        removeImg = textRemove(img)

        return (removeImg,)  # Return value must be a tuple


# This dictionary will be imported in __init__.py for node registration
NODE_CLASS_MAPPINGS = {
    "ImageRemoveText": ImageRemoveTextNode
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRemoveText": "Image remove text"
}