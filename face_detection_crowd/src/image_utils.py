import cv2
import numpy as np
import os

def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Loaded image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image

def resize_image(image, width=None, height=None, max_dimension=None):
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image (numpy.ndarray): Input image.
        width (int, optional): Target width.
        height (int, optional): Target height.
        max_dimension (int, optional): Maximum dimension (width or height).
        
    Returns:
        numpy.ndarray: Resized image.
    """
    h, w = image.shape[:2]
    
    if max_dimension is not None:
        if w > h:
            width = max_dimension
            height = None
        else:
            width = None
            height = max_dimension
    
    if width is None and height is None:
        return image
    
    if width is None:
        aspect_ratio = height / float(h)
        new_width = int(w * aspect_ratio)
        new_height = height
    elif height is None:
        aspect_ratio = width / float(w)
        new_width = width
        new_height = int(h * aspect_ratio)
    else:
        new_width = width
        new_height = height
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized