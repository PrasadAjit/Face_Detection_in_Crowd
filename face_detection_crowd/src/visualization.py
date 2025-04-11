import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def draw_faces(image, faces, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes around detected faces.
    
    Args:
        image (numpy.ndarray): Input image.
        faces (list): List of (x, y, w, h) tuples representing face bounding boxes.
        color (tuple): BGR color for the bounding boxes.
        thickness (int): Line thickness.
        
    Returns:
        numpy.ndarray: Image with drawn bounding boxes.
    """
    img_copy = image.copy()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
    
    return img_copy

def draw_advanced_faces(image, faces, include_labels=True, include_ids=True, blur_faces=False):
    """
    Draw advanced visualizations for detected faces.
    
    Args:
        image (numpy.ndarray): Input image.
        faces (list): List of (x, y, w, h) tuples representing face bounding boxes.
        include_labels (bool): Whether to include face count labels.
        include_ids (bool): Whether to include face ID numbers.
        blur_faces (bool): Whether to blur the faces for privacy.
        
    Returns:
        numpy.ndarray: Image with visualizations.
    """
    img_copy = image.copy()
    
    # Generate random colors for face boxes
    colors = []
    for _ in range(len(faces)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    
    for i, (x, y, w, h) in enumerate(faces):
        # Draw bounding box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), colors[i], 2)
        
        if include_ids:
            # Draw face ID
            cv2.putText(img_copy, f"#{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, colors[i], 2)
        
        if blur_faces:
            # Apply blur to the face region
            face_region = img_copy[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (23, 23), 30)
            img_copy[y:y+h, x:x+w] = blurred_face
    
    if include_labels:
        # Add total face count at the top of the image
        label = f"Total Faces: {len(faces)}"
        cv2.putText(img_copy, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
    
    return img_copy

def display_image(image, title="Image"):
    """
    Display an image using matplotlib.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        title (str): Title for the image.
    """
    # Convert BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.show()