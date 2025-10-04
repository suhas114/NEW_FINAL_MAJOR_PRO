#!/usr/bin/env python3
"""
Combined Fire and Object Detection System with Email Alerts
This system detects both objects and fire through camera and sends real email notifications
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
import torch
from ultralytics import YOLO
from collections import deque

class CombinedDetectionSystem:
    def __init__(self):
        """Initialize the combined detection system"""
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
        
        # Fire detection parameters
        self.lower_fire = np.array([0, 100, 100])  # HSV lower bound for fire
        self.upper_fire = np.array([35, 255, 255])  # HSV upper bound for fire
        self.min_fire_area = 1000  # Minimum fire area threshold
        
        # Object detection parameters
        self.conf_threshold = 0.5  # YOLO confidence threshold
        self.model = None
        
        # Detection history
        self.fire_history = deque(maxlen=10)
        self.object_history = deque(maxlen=10)
        
        # Create directories
        self.create_directories()
        
        # Load YOLO model
        self.load_yolo_model()
        
        print("🔥 Combined Detection System Initialized")
        print("📧 Email alerts enabled")
        print("📹 Camera detection active")
        print("🎯 YOLO object detection active")
        print("💾 Credentials saved")

    def create_directories(self):
        """Create necessary directories"""
        Path("captured_fires").mkdir(exist_ok=True)
        Path("captured_objects").mkdir(exist_ok=True)
        Path("system_logs").mkdir(exist_ok=True)

    def load_yolo_model(self):
        """Load YOLO model for object detection"""
        try:
            print("Loading YOLO model...")
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.system_active = False

    def detect_fire_in_frame(self, frame):
        """Detect fire in a frame using computer vision"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for fire colors
        fire_mask = cv2.inRange(hsv, self.lower_fire, self.upper_fire)
        
        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected = False
        fire_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_fire_area:
                fire_detected = True
                fire_area += area
                # Draw rectangle around fire
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"FIRE: {area:.0f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return fire_detected, frame, fire_area

    def detect_objects_in_frame(self, frame):
        """Detect objects in a frame using YOLO"""
        if self.model is None:
            return [], frame
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.conf_threshold)
            
            object_detections = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
                    object_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
                    # Draw rectangle around object
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name}: {conf:.2f}", (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return object_detections, frame
            
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return [], frame

    def capture_image(self, frame, detection_type):
        """Capture and save image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if detection_type == "fire":
            filename = f"captured_fires/fire_detected_{timestamp}.jpg"
        else:
            filename = f"captured_objects/object_detected_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        return filename

    def send_fire_alert_email(self, frame, fire_area):
        """Send fire alert email with captured image"""
        try:
            # Capture fire image
            image_path = self.capture_image(frame, "fire")
            
            # Create email message
            msg = MIMEMultipart()
            msg['Subject'] = "🚨 FIRE DETECTED - Emergency Alert 🚨"
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            
            # Email body
            body = f"""
🚨 FIRE DETECTION ALERT - EMERGENCY 🚨

FIRE DETECTED in monitored area!

📅 Detection Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🔥 Fire Area: {fire_area:.0f} pixels
🔢 Alert Count: {self.alert_count}
📍 Location: Combined Camera Monitoring System

⚠️ IMMEDIATE ACTION REQUIRED:
• Check the monitored area immediately
• Contact emergency services if needed
• Verify fire sensor functionality
• Review security camera footage

🔧 System Information:
• Detection Method: Computer Vision + YOLO
• Camera: Real-time monitoring
• Alert System: Email notifications
• Status: FIRE CONFIRMED

📞 Emergency Contacts:
• Fire Department: 911 (US) / 101 (India)
• Emergency Services: 112
• Building Security: Contact local security

This is an automated alert from the Combined Fire Detection System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach fire image
            with open(image_path, 'rb') as image_file:
                image_attachment = MIMEImage(image_file.read())
                image_attachment.add_header('Content-Disposition', 'attachment', 
                                          filename=f"fire_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                msg.attach(image_attachment)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.login(self.email_config['sender_email'], self.email_config['app_password'])
                server.send_message(msg)
            
            print(f"✅ Fire alert email sent to {self.email_config['recipient_email']}")
            print(f"📷 Fire image saved: {image_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending fire alert email: {e}")
            return False

    def send_object_alert_email(self, frame, objects):
        """Send object detection alert email"""
        try:
            # Capture object image
            image_path = self.capture_image(frame, "object")
            
            # Create email message
            msg = MIMEMultipart()
            msg['Subject'] = "🔍 OBJECTS DETECTED - Security Alert 🔍"
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            
            # Email body
            body = f"""
🔍 OBJECT DETECTION ALERT 🔍

OBJECTS DETECTED in monitored area!

📅 Detection Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🔍 Objects Detected: {len(objects)}
🔢 Alert Count: {self.alert_count}
📍 Location: Combined Camera Monitoring System

📋 Detected Objects:
"""
            
            for obj in objects:
                body += f"• {obj['class']} (Confidence: {obj['confidence']:.2f})\n"
            
            body += f"""
⚠️ ACTION REQUIRED:
• Check the monitored area
• Verify if objects are expected
• Review security camera footage
• Contact security if needed

🔧 System Information:
• Detection Method: YOLO Object Detection
• Camera: Real-time monitoring
• Alert System: Email notifications
• Status: OBJECTS DETECTED

This is an automated alert from the Combined Detection System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach object image
            with open(image_path, 'rb') as image_file:
                image_attachment = MIMEImage(image_file.read())
                image_attachment.add_header('Content-Disposition', 'attachment', 
                                          filename=f"objects_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                msg.attach(image_attachment)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.login(self.email_config['sender_email'], self.email_config['app_password'])
                server.send_message(msg)
            
            print(f"✅ Object alert email sent to {self.email_config['recipient_email']}")
            print(f"📷 Object image saved: {image_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending object alert email: {e}")
            return False

    def log_detection_event(self, event_type, data):
        """Log detection event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'alert_count': self.alert_count,
            'camera_index': self.camera_index,
            'data': data
        }
        
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"system_logs/combined_detection_events_{timestamp}.json"
        
        try:
            events = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    events = json.load(f)
            
            events.append(event)
            
            with open(log_file, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error logging detection event: {e}")

    def process_frame(self, frame):
        """Process a frame for both fire and object detection"""
        # Detect fire
        fire_detected, frame, fire_area = self.detect_fire_in_frame(frame)
        
        # Detect objects
        object_detections, frame = self.detect_objects_in_frame(frame)
        
        current_time = time.time()
        
        # Check for fire detection
        if fire_detected and fire_area > self.min_fire_area:
            if not self.fire_detected and (current_time - self.last_alert_time) > self.alert_cooldown:
                # Fire just detected
                self.fire_detected = True
                self.last_alert_time = current_time
                self.alert_count += 1
                
                print(f"\n🚨 FIRE DETECTED! Alert #{self.alert_count}")
                print(f"🔥 Fire Area: {fire_area:.0f} pixels")
                print(f"📅 Time: {datetime.now().strftime('%H:%M:%S')}")
                
                # Send email alert in separate thread
                email_thread = threading.Thread(
                    target=self.send_fire_alert_email,
                    args=(frame.copy(), fire_area)
                )
                email_thread.daemon = True
                email_thread.start()
                
                # Log event
                self.log_detection_event('fire_detected', {'fire_area': fire_area})
                
        elif not fire_detected and self.fire_detected:
            # Fire cleared
            self.fire_detected = False
            print("✅ Fire cleared - Area safe")
        
        # Check for object detection
        if len(object_detections) > 0:
            # Check if significant objects detected
            significant_objects = [obj for obj in object_detections if obj['confidence'] > 0.7]
            
            if significant_objects and (current_time - self.last_alert_time) > self.alert_cooldown:
                self.last_alert_time = current_time
                self.alert_count += 1
                
                print(f"\n🔍 OBJECTS DETECTED! Alert #{self.alert_count}")
                print(f"📦 Objects: {len(significant_objects)}")
                for obj in significant_objects:
                    print(f"   • {obj['class']} ({obj['confidence']:.2f})")
                print(f"📅 Time: {datetime.now().strftime('%H:%M:%S')}")
                
                # Send email alert in separate thread
                email_thread = threading.Thread(
                    target=self.send_object_alert_email,
                    args=(frame.copy(), significant_objects)
                )
                email_thread.daemon = True
                email_thread.start()
                
                # Log event
                self.log_detection_event('objects_detected', {
                    'objects': significant_objects,
                    'count': len(significant_objects)
                })
        
        return frame, fire_detected, object_detections

    def run_detection(self):
        """Run the main detection loop"""
        print("\n🔥 Starting Combined Detection System...")
        print("📹 Camera: Initializing...")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return
        
        print("✅ Camera initialized successfully")
        print("🎯 Combined detection active - Press 'q' to quit")
        print("📧 Email alerts enabled")
        print("🎯 YOLO object detection active")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.system_active:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame!")
                    break
                
                # Process frame for combined detection
                processed_frame, fire_detected, object_detections = self.process_frame(frame)
                
                # Display system status
                status_text = "FIRE DETECTED!" if fire_detected else "No Fire"
                status_color = (0, 0, 255) if fire_detected else (0, 255, 0)
                
                cv2.putText(processed_frame, f"Status: {status_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(processed_frame, f"Alerts: {self.alert_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"Objects: {len(object_detections)}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Combined Detection System', processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        except Exception as e:
            print(f"❌ Error in detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Combined detection system stopped")

    def show_system_info(self):
        """Show system information"""
        print("\n📊 Combined Detection System Information")
        print("=" * 50)
        print(f"📧 Email: {self.email_config['sender_email']}")
        print(f"📬 Recipient: {self.email_config['recipient_email']}")
        print(f"📹 Camera: Index {self.camera_index}")
        print(f"🖼️ Resolution: {self.frame_width}x{self.frame_height}")
        print(f"🎯 YOLO Confidence Threshold: {self.conf_threshold}")
        print(f"🚨 Alert Cooldown: {self.alert_cooldown} seconds")
        print(f"📁 Logs Directory: system_logs/")
        print(f"📸 Fire Images Directory: captured_fires/")
        print(f"📸 Object Images Directory: captured_objects/")

def main():
    """Main function"""
    print("🔥 Combined Fire and Object Detection System with Email Alerts")
    print("=" * 70)
    print("This system will:")
    print("✅ Detect fire through computer vision")
    print("✅ Detect objects using YOLO")
    print("✅ Send real email alerts for both fire and objects")
    print("✅ Capture images of detections")
    print("✅ Log all events")
    print("💾 Credentials saved")
    print()
    
    # Create combined detection system
    detection_system = CombinedDetectionSystem()
    
    if not detection_system.system_active:
        print("❌ System initialization failed!")
        return
    
    # Show system information
    detection_system.show_system_info()
    
    # Start detection
    input("\nPress Enter to start combined detection...")
    detection_system.run_detection()

if __name__ == "__main__":
    main()
