import argparse
import cv2
import os
import time
from src.face_detector import FaceDetector
from src.image_utils import load_image, resize_image
from src.visualization import draw_faces, draw_advanced_faces, display_image

def main():
    """
    Main function to run face detection on an image.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Detection in Crowds')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--method', type=str, default='haar', 
                        choices=['haar', 'dnn', 'hog'], help='Face detection method')
    parser.add_argument('--output', type=str, help='Path to save output image')
    parser.add_argument('--visualize', action='store_true', help='Display visualization')
    parser.add_argument('--advanced', action='store_true', help='Use advanced visualization')
    parser.add_argument('--blur', action='store_true', help='Blur detected faces for privacy')
    
    args = parser.parse_args()
    
    # Initialize face detector
    detector = FaceDetector(method=args.method)
    
    # Load and resize image
    image = load_image(args.image)
    resized_image = resize_image(image, max_dimension=1200)
    
    # Detect faces
    start_time = time.time()
    faces = detector.detect_faces(resized_image)
    end_time = time.time()
    
    # Print results
    print(f"Detected {len(faces)} faces in {end_time - start_time:.2f} seconds")
    
    # Visualize results
    if args.advanced:
        result_image = draw_advanced_faces(resized_image, faces, blur_faces=args.blur)
    else:
        result_image = draw_faces(resized_image, faces)
    
    # Save output if requested
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(args.output, result_image)
        print(f"Output saved to {args.output}")
    
    # Display image if requested
    if args.visualize:
        display_image(result_image, f"{len(faces)} faces detected")
    
if __name__ == "__main__":
    main()