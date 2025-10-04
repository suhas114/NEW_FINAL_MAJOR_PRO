#!/usr/bin/env python3
"""
YOLOv8 Fire Detection System
This system uses YOLOv8 with a trained fire detection model for accurate fire detection
"""

import cv2
import numpy as np
import time
import threading
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import os
import json
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics"])
    from ultralytics import YOLO

class YOLOv8FireDetection:
    def __init__(self, model_path=None):
        """Initialize the YOLOv8 fire detection system"""
        self.fire_detected = False
        self.alert_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 30  # seconds between alerts
        self.system_active = True
        
        # Email configuration
        self.email_config = {
            'sender_email': 'suhas123kichcha@gmail.com',
            'app_password': 'baka yybc wcbj kawv',
            'recipient_email': 'suhas123kichcha@gmail.com'
        }
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
        # YOLOv8 detection parameters
        self.confidence_threshold = 0.5  # Minimum confidence for detection
        self.iou_threshold = 0.45        # Non-maximum suppression threshold
        self.min_fire_area = 1000        # Minimum fire area in pixels
        self.max_fire_area = 100000      # Maximum fire area
        
        # Object detection settings
        self.enable_object_detection = True  # Enable general object detection
        self.object_detection_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'dog', 'cat']  # Classes to detect
        self.min_object_area = 500       # Minimum object area in pixels
        
        # Detection history for stability
        self.detection_history = deque(maxlen=10)
        self.fire_history = deque(maxlen=15)
        
        # Create directories
        self.create_directories()
        
        # Load YOLOv8 model
        self.load_model(model_path)
        
        # Load email configuration
        self.load_email_config()
        
        print("🔥 YOLOv8 Fire & Object Detection System Initialized")
        print(f"📊 Model: {self.model_path}")
        print(f"🎯 Confidence Threshold: {self.confidence_threshold}")
        print(f"📦 Object Detection: {'Enabled' if self.enable_object_detection else 'Disabled'}")
        print("📧 Email alerts enabled")
        print("📹 Real-time detection active")

    def create_directories(self):
        """Create necessary directories"""
        Path("captured_fires").mkdir(exist_ok=True)
        Path("system_logs").mkdir(exist_ok=True)

    def load_model(self, model_path=None):
        """Load YOLOv8 model for fire detection"""
        try:
            # Try to use the trained fire detection model first
            if model_path is None:
                # Check for trained models in order of preference
                possible_models = [
                    "runs/detect/forest_fire_model6/weights/best.pt",
                    "runs/detect/forest_fire_model2/weights/best.pt",
                    "yolov8n.pt"  # Fallback to base model
                ]
                
                for model in possible_models:
                    if os.path.exists(model):
                        model_path = model
                        break
            
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.model_path = model_path
                print(f"✅ Loaded model: {model_path}")
            else:
                # Use base YOLOv8 model
                self.model = YOLO('yolov8n.pt')
                self.model_path = 'yolov8n.pt'
                print("⚠️  Using base YOLOv8 model (not trained for fire detection)")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("📥 Downloading YOLOv8 base model...")
            self.model = YOLO('yolov8n.pt')
            self.model_path = 'yolov8n.pt'

    def load_email_config(self):
        """Load email configuration"""
        print("\n📧 Email Configuration")
        print("=" * 40)
        
        # Email credentials are already set in __init__
        if self.email_config['sender_email'] and self.email_config['app_password']:
            print(f"✅ Email configured: {self.email_config['sender_email']}")
            print("📧 Email alerts enabled")
        else:
            print("❌ Email configuration incomplete! Alerts disabled.")
            self.email_config['sender_email'] = None

    def detect_fire_yolov8(self, frame):
        """Detect fire using YOLOv8 model"""
        try:
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
            
            fire_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = result.names[class_id] if hasattr(result, 'names') and class_id in result.names else f"class_{class_id}"
                        
                        # Check if it's fire or smoke detection
                        if 'fire' in class_name.lower() or 'smoke' in class_name.lower() or class_id in [0, 1]:  # Assuming fire=0, smoke=1
                            area = (x2 - x1) * (y2 - y1)
                            
                            if self.min_fire_area <= area <= self.max_fire_area:
                                fire_detections.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': float(confidence),
                                    'class_name': class_name,
                                    'area': area
                                })
            
            return fire_detections
            
        except Exception as e:
            print(f"❌ Error in YOLOv8 detection: {e}")
            return []

    def detect_objects_yolov8(self, frame):
        """Detect general objects using YOLOv8 model"""
        try:
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
            
            object_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = result.names[class_id] if hasattr(result, 'names') and class_id in result.names else f"class_{class_id}"
                        
                        # Check if it's an object we want to detect
                        if (self.enable_object_detection and 
                            any(obj_class in class_name.lower() for obj_class in self.object_detection_classes)):
                            area = (x2 - x1) * (y2 - y1)
                            
                            if area >= self.min_object_area:
                                object_detections.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': float(confidence),
                                    'class_name': class_name,
                                    'area': area
                                })
            
            return object_detections
            
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return []

    def analyze_detection_stability(self, detections):
        """Analyze detection stability to reduce false positives"""
        if not detections:
            self.detection_history.append(False)
            return False
        
        # Add current detection to history
        self.detection_history.append(True)
        
        # Check if we have enough recent detections
        recent_detections = list(self.detection_history)[-5:]  # Last 5 frames
        detection_rate = sum(recent_detections) / len(recent_detections)
        
        return detection_rate >= 0.6  # At least 60% detection rate in recent frames

    def draw_detections(self, frame, fire_detections, object_detections=None):
        """Draw detection boxes and labels on frame"""
        # Draw fire detections
        for detection in fire_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            color = (0, 0, 255) if 'fire' in class_name.lower() else (0, 255, 255)  # Red for fire, yellow for smoke
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw object detections
        if object_detections:
            for detection in object_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Draw bounding box (blue for objects)
                color = (255, 0, 0)  # Blue for objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

    def save_fire_image(self, frame, fire_detections, object_detections=None):
        """Save image with fire and object detections"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_fires/yolov8_fire_{timestamp}.jpg"
        
        # Draw detections on the saved image
        annotated_frame = self.draw_detections(frame.copy(), fire_detections, object_detections)
        cv2.imwrite(filename, annotated_frame)
        
        return filename

    def send_email_alert(self, image_path, fire_detections, object_detections=None):
        """Send email alert with fire and object detection image"""
        if not self.email_config['sender_email'] or not self.email_config['app_password']:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = "🚨 FIRE DETECTED - YOLOv8 Alert"
            
            # Email body
            body = f"""
🚨 FIRE DETECTION ALERT 🚨

Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Location: Camera Feed
Detection Method: YOLOv8 Fire & Object Detection System

🔥 Fire Detections:
"""
            for i, detection in enumerate(fire_detections, 1):
                body += f"{i}. {detection['class_name']} (Confidence: {detection['confidence']:.2f})\n"
            
            if object_detections:
                body += "\n📦 Object Detections:\n"
                for i, detection in enumerate(object_detections, 1):
                    body += f"{i}. {detection['class_name']} (Confidence: {detection['confidence']:.2f})\n"
            
            body += "\n⚠️  IMMEDIATE ACTION REQUIRED ⚠️"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                msg.attach(img)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.login(self.email_config['sender_email'], self.email_config['app_password'])
                server.send_message(msg)
            
            print(f"📧 Email alert sent successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return False

    def log_detection_event(self, fire_detections, object_detections=None, image_path=None):
        """Log detection event to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"system_logs/yolov8_events_{timestamp}.json"
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'fire_detections': fire_detections,
            'object_detections': object_detections or [],
            'image_path': image_path,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold
        }
        
        # Load existing logs or create new
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(event)
        
        # Save logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def process_frame(self, frame):
        """Process a single frame for fire and object detection"""
        # Detect fire using YOLOv8
        fire_detections = self.detect_fire_yolov8(frame)
        
        # Detect objects using YOLOv8
        object_detections = self.detect_objects_yolov8(frame) if self.enable_object_detection else []
        
        # Analyze detection stability
        stable_detection = self.analyze_detection_stability(fire_detections)
        
        current_time = time.time()
        
        if stable_detection and fire_detections:
            # Check if enough time has passed since last alert
            if current_time - self.last_alert_time > self.alert_cooldown:
                self.fire_detected = True
                self.alert_count += 1
                self.last_alert_time = current_time
                
                # Save image
                image_path = self.save_fire_image(frame, fire_detections, object_detections)
                
                # Log event
                self.log_detection_event(fire_detections, object_detections, image_path)
                
                # Send email alert
                if self.email_config['sender_email']:
                    threading.Thread(target=self.send_email_alert, args=(image_path, fire_detections, object_detections)).start()
                
                print(f"🔥 FIRE DETECTED! Alert #{self.alert_count}")
                for detection in fire_detections:
                    print(f"   - {detection['class_name']}: {detection['confidence']:.2f}")
                
                if object_detections:
                    print(f"📦 Objects detected: {len(object_detections)}")
                    for detection in object_detections:
                        print(f"   - {detection['class_name']}: {detection['confidence']:.2f}")
        
        # Draw detections on frame
        annotated_frame = self.draw_detections(frame, fire_detections, object_detections)
        
        # Add status overlay
        status_text = f"YOLOv8 Fire & Object Detection - Alerts: {self.alert_count}"
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.fire_detected:
            cv2.putText(annotated_frame, "FIRE DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add object count
        if object_detections:
            object_text = f"Objects: {len(object_detections)}"
            cv2.putText(annotated_frame, object_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return annotated_frame

    def run_detection(self):
        """Run the main fire detection loop"""
        print("🎥 Starting camera...")
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera started successfully")
        print("🎯 Press 'q' to quit, 's' to save current frame")
        
        try:
            while self.system_active:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('YOLOv8 Fire Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_fires/manual_save_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Frame saved: {filename}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🔚 Fire detection system stopped")

def main():
    """Main function to run the YOLOv8 fire detection system"""
    print("🔥 YOLOv8 Fire Detection System")
    print("=" * 50)
    
    # Initialize detection system
    detector = YOLOv8FireDetection()
    
    # Run detection
    detector.run_detection()

if __name__ == "__main__":
    main()
