#!/usr/bin/env python3
"""
Test YOLOv8 Fire Detection System
This script tests the YOLOv8 fire detection with sample images
"""

import cv2
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolov8_fire_detection import YOLOv8FireDetection

def test_model_loading():
    """Test if the YOLOv8 model loads correctly"""
    print("🧪 Testing YOLOv8 model loading...")
    
    try:
        detector = YOLOv8FireDetection()
        print(f"✅ Model loaded successfully: {detector.model_path}")
        return detector
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_image_detection(detector, image_path):
    """Test fire detection on a single image"""
    print(f"🧪 Testing detection on: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return False
        
        # Run fire detection
        fire_detections = detector.detect_fire_yolov8(image)
        
        # Run object detection
        object_detections = detector.detect_objects_yolov8(image) if detector.enable_object_detection else []
        
        print(f"🔥 Fire detections found: {len(fire_detections)}")
        for i, detection in enumerate(fire_detections, 1):
            print(f"   {i}. {detection['class_name']}: {detection['confidence']:.3f}")
        
        if object_detections:
            print(f"📦 Object detections found: {len(object_detections)}")
            for i, detection in enumerate(object_detections, 1):
                print(f"   {i}. {detection['class_name']}: {detection['confidence']:.3f}")
        
        # Draw detections
        annotated_image = detector.draw_detections(image, fire_detections, object_detections)
        
        # Save annotated image
        output_path = f"test_output_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, annotated_image)
        print(f"💾 Annotated image saved: {output_path}")
        
        return len(fire_detections) > 0 or len(object_detections) > 0
        
    except Exception as e:
        print(f"❌ Error testing image: {e}")
        return False

def test_camera_feed():
    """Test camera feed detection"""
    print("🧪 Testing camera feed...")
    
    try:
        detector = YOLOv8FireDetection()
        
        # Test camera for a few seconds
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        print("📹 Camera opened successfully. Press 'q' to stop test.")
        
        frame_count = 0
        max_frames = 100  # Test for 100 frames
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = detector.process_frame(frame)
            
            # Display frame
            cv2.imshow('YOLOv8 Test', processed_frame)
            
            # Check for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing camera: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 YOLOv8 Fire & Object Detection Test Suite")
    print("=" * 50)
    
    # Test 1: Model loading
    detector = test_model_loading()
    if detector is None:
        print("❌ Model loading failed. Exiting.")
        return
    
    # Test 2: Sample images (if available)
    sample_images = [
        "captured_fires/fire_detected_20250830_094714.jpg",
        "demo_fires/demo_fire_20250829_212612.jpg"
    ]
    
    for image_path in sample_images:
        if os.path.exists(image_path):
            test_image_detection(detector, image_path)
            print()
    
    # Test 3: Camera feed
    print("🎥 Testing camera feed (5 seconds)...")
    test_camera_feed()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
