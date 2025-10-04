#!/usr/bin/env python3
"""
Fire Detection System with Camera and Email Alerts
This system detects fire through camera and sends real email notifications
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

class FireDetectionSystem:
    def __init__(self):
        """Initialize the fire detection system"""
        self.fire_detected = False
        self.alert_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 30  # seconds between alerts
        self.system_active = True
        
        # Email configuration
        self.email_config = {
            'sender_email': None,
            'app_password': None,
            'recipient_email': 'suhas123kichcha@gmail.com'
        }
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
        # Fire detection parameters
        self.lower_fire = np.array([0, 100, 100])  # HSV lower bound for fire
        self.upper_fire = np.array([35, 255, 255])  # HSV upper bound for fire
        
        # Create directories
        self.create_directories()
        
        # Load email configuration
        self.load_email_config()
        
        print("🔥 Fire Detection System Initialized")
        print("📧 Email alerts enabled")
        print("📹 Camera detection active")

    def create_directories(self):
        """Create necessary directories"""
        Path("captured_fires").mkdir(exist_ok=True)
        Path("system_logs").mkdir(exist_ok=True)

    def load_email_config(self):
        """Load email configuration from user input"""
        print("\n📧 Email Configuration Setup")
        print("=" * 40)
        
        # Get Gmail credentials
        self.email_config['sender_email'] = input("Enter your Gmail address: ").strip()
        
        print("\n🔧 Gmail App Password Setup:")
        print("1. Go to myaccount.google.com")
        print("2. Security → 2-Step Verification → Get started")
        print("3. Enable 2-Factor Authentication")
        print("4. Go to Security → App passwords")
        print("5. Select 'Mail' as app")
        print("6. Generate password (16 characters)")
        print()
        
        self.email_config['app_password'] = input("Enter your Gmail App Password (16 chars): ").strip()
        
        if not all([self.email_config['sender_email'], self.email_config['app_password']]):
            print("❌ Email configuration incomplete!")
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
            if area > 500:  # Minimum area threshold
                fire_detected = True
                fire_area += area
                # Draw rectangle around fire
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"FIRE: {area:.0f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return fire_detected, frame, fire_area

    def capture_fire_image(self, frame):
        """Capture and save fire image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_fires/fire_detected_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

    def send_fire_alert_email(self, frame, fire_area):
        """Send fire alert email with captured image"""
        try:
            # Capture fire image
            image_path = self.capture_fire_image(frame)
            
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
📍 Location: Camera Monitoring System

⚠️ IMMEDIATE ACTION REQUIRED:
• Check the monitored area immediately
• Contact emergency services if needed
• Verify fire sensor functionality
• Review security camera footage

🔧 System Information:
• Detection Method: Computer Vision
• Camera: Real-time monitoring
• Alert System: Email notifications
• Status: FIRE CONFIRMED

📞 Emergency Contacts:
• Fire Department: 911 (US) / 101 (India)
• Emergency Services: 112
• Building Security: Contact local security

This is an automated alert from the Fire Detection System.
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

    def log_fire_event(self, fire_area):
        """Log fire detection event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'fire_detected',
            'alert_count': self.alert_count,
            'fire_area': fire_area,
            'camera_index': self.camera_index
        }
        
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"system_logs/fire_events_{timestamp}.json"
        
        try:
            events = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    events = json.load(f)
            
            events.append(event)
            
            with open(log_file, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error logging fire event: {e}")

    def process_frame(self, frame):
        """Process a frame for fire detection"""
        # Detect fire
        fire_detected, processed_frame, fire_area = self.detect_fire_in_frame(frame)
        
        current_time = time.time()
        
        if fire_detected and fire_area > 1000:  # Minimum area threshold
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
                    args=(processed_frame.copy(), fire_area)
                )
                email_thread.daemon = True
                email_thread.start()
                
                # Log event
                self.log_fire_event(fire_area)
                
        elif not fire_detected and self.fire_detected:
            # Fire cleared
            self.fire_detected = False
            print("✅ Fire cleared - Area safe")
        
        return processed_frame

    def run_fire_detection(self):
        """Run the main fire detection loop"""
        print("\n🔥 Starting Fire Detection System...")
        print("📹 Camera: Initializing...")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return
        
        print("✅ Camera initialized successfully")
        print("🎯 Fire detection active - Press 'q' to quit")
        print("📧 Email alerts enabled")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.system_active:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame!")
                    break
                
                # Process frame for fire detection
                processed_frame = self.process_frame(frame)
                
                # Display system status
                status_text = "FIRE DETECTED!" if self.fire_detected else "No Fire"
                status_color = (0, 0, 255) if self.fire_detected else (0, 255, 0)
                
                cv2.putText(processed_frame, f"Status: {status_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(processed_frame, f"Alerts: {self.alert_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Fire Detection System', processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Fire detection stopped by user")
        except Exception as e:
            print(f"❌ Error in fire detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Fire detection system stopped")

    def show_system_info(self):
        """Show system information"""
        print("\n📊 Fire Detection System Information")
        print("=" * 40)
        print(f"📧 Email: {self.email_config['sender_email']}")
        print(f"📬 Recipient: {self.email_config['recipient_email']}")
        print(f"📹 Camera: Index {self.camera_index}")
        print(f"🖼️ Resolution: {self.frame_width}x{self.frame_height}")
        print(f"🚨 Alert Cooldown: {self.alert_cooldown} seconds")
        print(f"📁 Logs Directory: system_logs/")
        print(f"📸 Images Directory: captured_fires/")

def main():
    """Main function"""
    print("🔥 Fire Detection System with Camera and Email Alerts")
    print("=" * 60)
    print("This system will:")
    print("✅ Detect fire through camera")
    print("✅ Send real email alerts")
    print("✅ Capture fire images")
    print("✅ Log all events")
    print()
    
    # Create fire detection system
    fire_system = FireDetectionSystem()
    
    if not fire_system.system_active:
        print("❌ System initialization failed!")
        return
    
    # Show system information
    fire_system.show_system_info()
    
    # Start fire detection
    input("\nPress Enter to start fire detection...")
    fire_system.run_fire_detection()

if __name__ == "__main__":
    main()
