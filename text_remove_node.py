import os
from PIL import Image
import numpy as np

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
        pil_image = image.cpu().numpy().astype(np.float32)
        # Get image information
        removeImg = textRemove(pil_image)

        return (removeImg,)  # Return value must be a tuple


# This dictionary will be imported in __init__.py for node registration
NODE_CLASS_MAPPINGS = {
    "ImageRemoveText": ImageRemoveTextNode
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRemoveText": "Image remove text"
}