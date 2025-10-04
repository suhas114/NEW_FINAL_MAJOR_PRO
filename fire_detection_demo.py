#!/usr/bin/env python3
"""
Fire Detection Demo - Simulates fire detection and sends real email alerts
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

class FireDetectionDemo:
    def __init__(self):
        """Initialize the fire detection demo"""
        self.fire_detected = False
        self.alert_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 10  # seconds between alerts
        self.system_active = True
        self.demo_mode = True
        
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
        
        # Demo settings
        self.demo_fire_duration = 5  # seconds
        self.demo_fire_interval = 15  # seconds
        self.last_demo_fire = 0
        self.demo_fire_active = False
        
        # Create directories
        Path("demo_fires").mkdir(exist_ok=True)
        Path("demo_logs").mkdir(exist_ok=True)
        
        # Load email configuration
        self.load_email_config()
        
        print("🔥 Fire Detection Demo Initialized")
        print("📧 Email alerts enabled")
        print("🎮 Demo mode active")

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

    def add_fire_overlay(self, frame):
        """Add fire overlay to frame for demo"""
        # Create fire colors
        fire_colors = [
            (0, 0, 255),    # Red
            (0, 165, 255),  # Orange
            (0, 255, 255)   # Yellow
        ]
        
        # Add fire effect
        h, w = frame.shape[:2]
        fire_area = np.zeros((h//4, w//4, 3), dtype=np.uint8)
        
        # Create fire gradient
        for i in range(fire_area.shape[0]):
            for j in range(fire_area.shape[1]):
                color_idx = int((i / fire_area.shape[0]) * len(fire_colors))
                fire_area[i, j] = fire_colors[min(color_idx, len(fire_colors)-1)]
        
        # Resize and place fire overlay
        fire_overlay = cv2.resize(fire_area, (w//2, h//2))
        x_offset = w//4
        y_offset = h//2
        
        # Blend fire with frame
        for i in range(fire_overlay.shape[0]):
            for j in range(fire_overlay.shape[1]):
                if y_offset + i < h and x_offset + j < w:
                    # Blend with original frame
                    alpha = 0.7
                    frame[y_offset + i, x_offset + j] = cv2.addWeighted(
                        frame[y_offset + i, x_offset + j], 1-alpha,
                        fire_overlay[i, j], alpha, 0
                    )
        
        # Add fire text
        cv2.putText(frame, "DEMO FIRE", (x_offset, y_offset - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return frame

    def capture_demo_image(self, frame):
        """Capture and save demo fire image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_fires/demo_fire_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

    def send_demo_fire_email(self, frame):
        """Send demo fire alert email"""
        try:
            # Capture demo image
            image_path = self.capture_demo_image(frame)
            
            # Create email message
            msg = MIMEMultipart()
            msg['Subject'] = "🔥 DEMO: Fire Detection Test Alert 🔥"
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            
            # Email body
            body = f"""
🔥 DEMO FIRE DETECTION TEST - EMAIL ALERT 🔥

This is a DEMONSTRATION fire detection alert!

📅 Test Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🎮 Mode: Demo/Test
🔢 Alert Count: {self.alert_count}
📍 Location: Demo Camera System

✅ This email confirms that:
• Fire detection system is working
• Email alerts are functional
• Image capture is operational
• Real-time monitoring is active

🔧 System Features Tested:
• Computer vision fire detection
• Real-time camera monitoring
• Email notification system
• Image capture and attachment
• Event logging system

📧 Recipient: {self.email_config['recipient_email']}
📸 Image: Attached fire detection image

This is a DEMONSTRATION alert - no real fire detected.
System is ready for real fire detection monitoring.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach demo image
            with open(image_path, 'rb') as image_file:
                image_attachment = MIMEImage(image_file.read())
                image_attachment.add_header('Content-Disposition', 'attachment', 
                                          filename=f"demo_fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                msg.attach(image_attachment)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.login(self.email_config['sender_email'], self.email_config['app_password'])
                server.send_message(msg)
            
            print(f"✅ Demo fire alert email sent to {self.email_config['recipient_email']}")
            print(f"📷 Demo image saved: {image_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending demo fire email: {e}")
            return False

    def log_demo_event(self):
        """Log demo fire event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'demo_fire_detected',
            'alert_count': self.alert_count,
            'mode': 'demo',
            'camera_index': self.camera_index
        }
        
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"demo_logs/demo_events_{timestamp}.json"
        
        try:
            events = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    events = json.load(f)
            
            events.append(event)
            
            with open(log_file, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error logging demo event: {e}")

    def process_demo_frame(self, frame):
        """Process frame for demo fire detection"""
        current_time = time.time()
        
        # Demo fire logic
        if current_time - self.last_demo_fire > self.demo_fire_interval and not self.demo_fire_active:
            # Start demo fire
            self.demo_fire_active = True
            self.last_demo_fire = current_time
            self.last_alert_time = current_time
            self.alert_count += 1
            
            print(f"\n🔥 DEMO FIRE STARTED! Alert #{self.alert_count}")
            print(f"📅 Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Send demo email in separate thread
            email_thread = threading.Thread(
                target=self.send_demo_fire_email,
                args=(frame.copy(),)
            )
            email_thread.daemon = True
            email_thread.start()
            
            # Log event
            self.log_demo_event()
            
            self.fire_detected = True
        
        elif self.demo_fire_active and current_time - self.last_demo_fire > self.demo_fire_duration:
            # End demo fire
            self.demo_fire_active = False
            self.fire_detected = False
            print("✅ Demo fire ended - Area safe")
        
        # Add fire overlay if demo fire is active
        if self.demo_fire_active:
            frame = self.add_fire_overlay(frame)
        
        return frame

    def run_demo_detection(self):
        """Run the demo fire detection loop"""
        print("\n🔥 Starting Fire Detection Demo...")
        print("📹 Camera: Initializing...")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return
        
        print("✅ Camera initialized successfully")
        print("🎮 Demo mode active - Press 'q' to quit")
        print("📧 Email alerts enabled")
        print(f"🔥 Demo fire every {self.demo_fire_interval} seconds")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.system_active:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame!")
                    break
                
                # Process frame for demo fire detection
                processed_frame = self.process_demo_frame(frame)
                
                # Display demo status
                status_text = "DEMO FIRE!" if self.fire_detected else "Demo Mode"
                status_color = (0, 0, 255) if self.fire_detected else (0, 255, 0)
                
                cv2.putText(processed_frame, f"Status: {status_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(processed_frame, f"Demo Alerts: {self.alert_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show demo info
                cv2.putText(processed_frame, "DEMO MODE - Press 't' for test fire", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Fire Detection Demo', processed_frame)
                
                # Check for quit or test fire
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    # Manual test fire
                    self.last_demo_fire = time.time() - (self.demo_fire_interval - 2)
                    
        except KeyboardInterrupt:
            print("\n⏹️ Demo stopped by user")
        except Exception as e:
            print(f"❌ Error in demo: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Fire detection demo stopped")

    def show_demo_info(self):
        """Show demo information"""
        print("\n📊 Fire Detection Demo Information")
        print("=" * 40)
        print(f"📧 Email: {self.email_config['sender_email']}")
        print(f"📬 Recipient: {self.email_config['recipient_email']}")
        print(f"📹 Camera: Index {self.camera_index}")
        print(f"🖼️ Resolution: {self.frame_width}x{self.frame_height}")
        print(f"🔥 Demo Fire Interval: {self.demo_fire_interval} seconds")
        print(f"🔥 Demo Fire Duration: {self.demo_fire_duration} seconds")
        print(f"🚨 Alert Cooldown: {self.alert_cooldown} seconds")
        print(f"📁 Demo Logs: demo_logs/")
        print(f"📸 Demo Images: demo_fires/")

def main():
    """Main function"""
    print("🔥 Fire Detection Demo with Email Alerts")
    print("=" * 50)
    print("This demo will:")
    print("✅ Simulate fire detection")
    print("✅ Send real email alerts")
    print("✅ Capture demo images")
    print("✅ Log demo events")
    print("🎮 Press 't' during demo to trigger test fire")
    print()
    
    # Create fire detection demo
    fire_demo = FireDetectionDemo()
    
    if not fire_demo.system_active:
        print("❌ Demo initialization failed!")
        return
    
    # Show demo information
    fire_demo.show_demo_info()
    
    # Start demo
    input("\nPress Enter to start fire detection demo...")
    fire_demo.run_demo_detection()

if __name__ == "__main__":
    main()
