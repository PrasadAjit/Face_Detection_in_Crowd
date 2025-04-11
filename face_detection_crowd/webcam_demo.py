import cv2
import argparse
import time
from src.face_detector import FaceDetector
from src.visualization import draw_advanced_faces

def main():
    """
    Real-time face detection using webcam.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Face Detection')
    parser.add_argument('--method', type=str, default='haar', 
                       choices=['haar', 'dnn', 'hog'], help='Face detection method')
    parser.add_argument('--blur', action='store_true', help='Blur detected faces for privacy')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    
    args = parser.parse_args()
    
    # Initialize face detector
    detector = FaceDetector(method=args.method)
    
    # Open video capture
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Get window properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera resolution: {width}x{height}")
    print("Press 'q' to quit, 's' to save a screenshot")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Update FPS calculation
        frame_count += 1
        if frame_count >= 10:
            current_time = time.time()
            fps = frame_count / (current_time - fps_start_time)
            fps_start_time = current_time
            frame_count = 0
        
        # Process frame (every other frame to improve performance)
        if frame_count % 2 == 0:
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw faces with advanced visualization
            result_frame = draw_advanced_faces(frame, faces, blur_faces=args.blur)
            
            # Add FPS information
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Face Detection', result_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"Screenshot saved as {filename}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()