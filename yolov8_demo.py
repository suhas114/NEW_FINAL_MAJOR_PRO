#!/usr/bin/env python3
"""
YOLOv8 Fire Detection Demo
Quick demo of the YOLOv8 fire detection system without email setup
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolov8_fire_detection import YOLOv8FireDetection

class YOLOv8Demo:
    def __init__(self, model_path=None):
        """Initialize the demo system"""
        self.detector = YOLOv8FireDetection(model_path)
        
        # Disable email for demo
        self.detector.email_config['sender_email'] = None
        
        # Reduce alert cooldown for demo
        self.detector.alert_cooldown = 5  # 5 seconds between alerts
        
        print("🎬 YOLOv8 Fire & Object Detection Demo")
        print("=" * 40)
        print(f"📊 Model: {self.detector.model_path}")
        print(f"🎯 Confidence: {self.detector.confidence_threshold}")
        print(f"📦 Object Detection: {'Enabled' if self.detector.enable_object_detection else 'Disabled'}")
        print("📧 Email alerts: DISABLED (demo mode)")
        print("⏱️  Alert cooldown: 5 seconds")

    def run_demo(self):
        """Run the demo detection"""
        print("\n🎥 Starting camera demo...")
        print("🎮 Controls:")
        print("   'q' - Quit demo")
        print("   's' - Save current frame")
        print("   't' - Test with sample image")
        print("   'c' - Change confidence threshold")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera started successfully")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                # Process frame
                processed_frame = self.detector.process_frame(frame)
                
                # Add demo overlay
                cv2.putText(processed_frame, "DEMO MODE", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display frame
                cv2.imshow('YOLOv8 Fire & Object Detection Demo', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_fires/demo_save_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Frame saved: {filename}")
                elif key == ord('t'):
                    self.test_sample_images()
                elif key == ord('c'):
                    self.change_confidence()
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🔚 Demo ended")

    def test_sample_images(self):
        """Test detection on sample images"""
        print("\n🧪 Testing sample images...")
        
        sample_images = [
            "captured_fires/fire_detected_20250830_094714.jpg",
            "demo_fires/demo_fire_20250829_212612.jpg"
        ]
        
        for image_path in sample_images:
            if os.path.exists(image_path):
                print(f"📸 Testing: {image_path}")
                
                # Load and process image
                image = cv2.imread(image_path)
                if image is not None:
                    detections = self.detector.detect_fire_yolov8(image)
                    
                    if detections:
                        print(f"   🔥 Found {len(detections)} detection(s):")
                        for i, detection in enumerate(detections, 1):
                            print(f"      {i}. {detection['class_name']}: {detection['confidence']:.3f}")
                    else:
                        print("   ❌ No detections found")
                    
                    # Save annotated image
                    annotated = self.detector.draw_detections(image, detections)
                    output_path = f"test_output_{os.path.basename(image_path)}"
                    cv2.imwrite(output_path, annotated)
                    print(f"   💾 Saved: {output_path}")
                else:
                    print(f"   ❌ Could not load image")
            else:
                print(f"   ❌ Image not found: {image_path}")
        
        print("✅ Sample image test completed")

    def change_confidence(self):
        """Change confidence threshold"""
        print(f"\n🎯 Current confidence: {self.detector.confidence_threshold}")
        try:
            new_conf = float(input("Enter new confidence (0.1-1.0): "))
            if 0.1 <= new_conf <= 1.0:
                self.detector.confidence_threshold = new_conf
                print(f"✅ Confidence set to: {new_conf}")
            else:
                print("❌ Invalid confidence value (use 0.1-1.0)")
        except ValueError:
            print("❌ Invalid input")

def main():
    """Main demo function"""
    print("🔥 YOLOv8 Fire Detection Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = YOLOv8Demo()
    
    # Run demo
    demo.run_demo()

if __name__ == "__main__":
    main()
