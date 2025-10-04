#!/usr/bin/env python3
"""
YOLOv8 Fire Detection Runner
Simple script to run the YOLOv8 fire detection system
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Main function to run YOLOv8 fire detection"""
    parser = argparse.ArgumentParser(description='YOLOv8 Fire Detection System')
    parser.add_argument('--model', type=str, help='Path to YOLOv8 model (.pt file)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--no-email', action='store_true', help='Disable email alerts')
    
    args = parser.parse_args()
    
    print("🔥 YOLOv8 Fire Detection System")
    print("=" * 50)
    
    if args.test:
        print("🧪 Running in test mode...")
        from test_yolov8_fire_detection import main as test_main
        test_main()
        return
    
    # Import the main detection class
    try:
        from yolov8_fire_detection import YOLOv8FireDetection
    except ImportError as e:
        print(f"❌ Error importing YOLOv8FireDetection: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install ultralytics opencv-python numpy torch")
        return
    
    # Initialize detection system
    detector = YOLOv8FireDetection(model_path=args.model)
    
    # Override settings if provided
    if args.confidence != 0.5:
        detector.confidence_threshold = args.confidence
        print(f"🎯 Confidence threshold set to: {args.confidence}")
    
    if args.camera != 0:
        detector.camera_index = args.camera
        print(f"📹 Camera index set to: {args.camera}")
    
    if args.no_email:
        detector.email_config['sender_email'] = None
        print("📧 Email alerts disabled")
    
    # Run detection
    detector.run_detection()

if __name__ == "__main__":
    main()
