import cv2
import numpy as np
import os

class FaceDetector:
    """
    A class to detect faces in images using various methods.
    """
    def __init__(self, model_path=None, method="haar", min_confidence=0.5):
        """
        Initialize the face detector.
        
        Args:
            model_path (str): Path to the model file.
            method (str): Detection method ('haar', 'dnn', or 'hog').
            min_confidence (float): Minimum confidence threshold for detection.
        """
        self.method = method
        self.min_confidence = min_confidence
        
        if model_path is None:
            # Use default model paths based on method
            if method == "haar":
                model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            elif method == "dnn":
                # For DNN-based model, we'd use a caffe model
                print("DNN model requires downloading external files. Defaulting to Haar Cascade.")
                self.method = "haar"
                model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        if self.method == "haar":
            self.face_cascade = cv2.CascadeClassifier(model_path)
            if self.face_cascade.empty():
                raise ValueError(f"Error loading Haar Cascade model from {model_path}")
        elif self.method == "dnn":
            # DNN implementation would go here
            pass
        elif self.method == "hog":
            # We would use dlib's HOG-based detector here
            # This requires dlib, which we're not using in this basic implementation
            pass
        else:
            raise ValueError(f"Unsupported detection method: {method}")
    
    def detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            list: List of (x, y, w, h) tuples representing face bounding boxes.
        """
        if self.method == "haar":
            # Convert to grayscale for Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
        
        # Placeholder for other methods
        return []
    
    def get_face_count(self, image):
        """
        Count faces in an image.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            int: Number of faces detected.
        """
        faces = self.detect_faces(image)
        return len(faces)