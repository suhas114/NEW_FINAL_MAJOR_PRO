#!/usr/bin/env python3
"""
Accurate Fire Detection System with Advanced Computer Vision
This system uses multiple detection methods to reduce false positives
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

class AccurateFireDetection:
    def __init__(self):
        """Initialize the accurate fire detection system"""
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
        
        # Advanced detection parameters
        self.confidence_threshold = 0.6  # Minimum confidence for fire detection
        self.min_fire_area = 2000  # Minimum fire area in pixels
        self.max_fire_area = 50000  # Maximum fire area (to avoid false positives)
        self.fire_history = deque(maxlen=10)  # Track fire detection history
        
        # Multiple HSV ranges for different fire conditions
        self.fire_ranges = [
            # Bright fire (red-orange)
            ((0, 150, 150), (35, 255, 255)),
            # Darker fire (deep red)
            ((0, 100, 100), (20, 255, 200)),
            # Yellow flames
            ((20, 100, 100), (40, 255, 255))
        ]
        
        # Create directories
        self.create_directories()
        
        # Load email configuration
        self.load_email_config()
        
        print("🔥 Accurate Fire Detection System Initialized")
        print("📧 Email alerts enabled")
        print("🎯 Advanced detection algorithms active")

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

    def analyze_color_distribution(self, frame, mask):
        """Analyze color distribution to improve accuracy"""
        # Calculate color statistics within the mask
        if np.sum(mask) == 0:
            return 0
        
        # Get pixels within the mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masked_pixels = hsv[mask > 0]
        
        if len(masked_pixels) == 0:
            return 0
        
        # Calculate mean HSV values
        mean_hue = np.mean(masked_pixels[:, 0])
        mean_sat = np.mean(masked_pixels[:, 1])
        mean_val = np.mean(masked_pixels[:, 2])
        
        # Fire typically has high saturation and value
        sat_score = mean_sat / 255.0
        val_score = mean_val / 255.0
        
        # Calculate fire-like color score
        color_score = (sat_score + val_score) / 2.0
        
        return color_score

    def analyze_shape_features(self, contour):
        """Analyze shape features to distinguish fire from other objects"""
        if len(contour) < 5:
            return 0
        
        # Calculate shape features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0
        
        # Circularity (fire tends to be irregular)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 1
        
        # Fire typically has irregular shapes (low circularity)
        # and can have varying aspect ratios
        shape_score = 1 - min(circularity, 1)  # Lower circularity = higher score
        
        return shape_score

    def detect_motion(self, prev_frame, current_frame, mask):
        """Detect motion within the fire region"""
        if prev_frame is None:
            return 0.5  # Neutral score for first frame
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to focus on fire region
        prev_masked = cv2.bitwise_and(prev_gray, prev_gray, mask=mask)
        curr_masked = cv2.bitwise_and(curr_gray, curr_gray, mask=mask)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_masked, curr_masked, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        if flow is None:
            return 0
        
        # Calculate motion magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_score = np.mean(magnitude)
        
        # Normalize motion score
        normalized_motion = min(motion_score / 10.0, 1.0)
        
        return normalized_motion

    def detect_fire_accurately(self, frame, prev_frame):
        """Advanced fire detection using multiple criteria"""
        best_fire_score = 0
        best_contour = None
        best_mask = None
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try multiple HSV ranges
        for lower, upper in self.fire_ranges:
            # Create mask for current range
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Check area constraints
                if area < self.min_fire_area or area > self.max_fire_area:
                    continue
                
                # Calculate multiple scores
                color_score = self.analyze_color_distribution(frame, mask)
                shape_score = self.analyze_shape_features(contour)
                motion_score = self.detect_motion(prev_frame, frame, mask)
                
                # Combined fire score
                fire_score = (color_score * 0.5 + shape_score * 0.3 + motion_score * 0.2)
                
                if fire_score > best_fire_score:
                    best_fire_score = fire_score
                    best_contour = contour
                    best_mask = mask
        
        return best_fire_score, best_contour, best_mask

    def validate_fire_detection(self, frame, contour, mask, fire_score):
        """Additional validation to reduce false positives"""
        if contour is None or fire_score < self.confidence_threshold:
            return False, 0
        
        # Additional checks
        
        # 1. Check if the region is too uniform (likely not fire)
        if mask is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masked_pixels = hsv[mask > 0]
            
            if len(masked_pixels) > 0:
                # Calculate standard deviation of hue (fire should have variation)
                hue_std = np.std(masked_pixels[:, 0])
                if hue_std < 10:  # Too uniform
                    return False, fire_score
        
        # 2. Check temporal consistency
        self.fire_history.append(fire_score)
        if len(self.fire_history) >= 3:
            recent_scores = list(self.fire_history)[-3:]
            avg_score = np.mean(recent_scores)
            
            # Fire should be consistent over time
            if avg_score < self.confidence_threshold * 0.8:
                return False, fire_score
        
        # 3. Check contour complexity
        if contour is not None:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if perimeter > 0 and area > 0:
                complexity = perimeter / np.sqrt(area)
                
                # Fire typically has complex edges
                if complexity < 8:  # Too simple
                    return False, fire_score
        
        return True, fire_score

    def process_frame(self, frame, prev_frame):
        """Process a frame for accurate fire detection"""
        # Detect fire using advanced algorithm
        fire_score, contour, mask = self.detect_fire_accurately(frame, prev_frame)
        
        # Validate fire detection
        is_valid_fire, final_score = self.validate_fire_detection(frame, contour, mask, fire_score)
        
        fire_detected = is_valid_fire and final_score >= self.confidence_threshold
        fire_area = cv2.contourArea(contour) if contour is not None else 0
        
        # Draw detection results
        processed_frame = frame.copy()
        if fire_detected and contour is not None:
            # Draw rectangle around fire
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(processed_frame, f"FIRE: {final_score:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        current_time = time.time()
        
        if fire_detected and final_score >= self.confidence_threshold:
            if not self.fire_detected and (current_time - self.last_alert_time) > self.alert_cooldown:
                # Fire just detected
                self.fire_detected = True
                self.last_alert_time = current_time
                self.alert_count += 1
                
                print(f"\n🚨 ACCURATE FIRE DETECTED! Alert #{self.alert_count}")
                print(f"🔥 Confidence Score: {final_score:.2f}")
                print(f"📐 Fire Area: {fire_area:.0f} pixels")
                print(f"📅 Time: {datetime.now().strftime('%H:%M:%S')}")
                
                # Send email alert in separate thread
                email_thread = threading.Thread(
                    target=self.send_fire_alert_email,
                    args=(processed_frame.copy(), fire_area, final_score)
                )
                email_thread.daemon = True
                email_thread.start()
                
                # Log event
                self.log_fire_event(fire_area, final_score)
                
        elif not fire_detected and self.fire_detected:
            # Fire cleared
            self.fire_detected = False
            print("✅ Fire cleared - Area safe")
        
        return processed_frame, fire_detected, final_score

    def capture_fire_image(self, frame):
        """Capture and save fire image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_fires/accurate_fire_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

    def send_fire_alert_email(self, frame, fire_area, confidence):
        """Send accurate fire alert email"""
        try:
            # Capture fire image
            image_path = self.capture_fire_image(frame)
            
            # Create email message
            msg = MIMEMultipart()
            msg['Subject'] = "🚨 ACCURATE FIRE DETECTED - Emergency Alert 🚨"
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            
            # Email body
            body = f"""
🚨 ACCURATE FIRE DETECTION ALERT - EMERGENCY 🚨

FIRE CONFIRMED in monitored area with high accuracy!

📅 Detection Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🎯 Confidence: {confidence:.2f} ({confidence*100:.1f}%)
🔥 Fire Area: {fire_area:.0f} pixels
🔢 Alert Count: {self.alert_count}
📍 Location: Advanced Camera Monitoring System

⚠️ IMMEDIATE ACTION REQUIRED:
• Check the monitored area immediately
• Contact emergency services if needed
• Verify fire sensor functionality
• Review security camera footage

🔧 Advanced System Information:
• Detection Method: Multi-criteria Computer Vision
• Accuracy: {confidence*100:.1f}% confidence
• Validation: Color, Shape, Motion Analysis
• Status: FIRE CONFIRMED WITH HIGH ACCURACY

📞 Emergency Contacts:
• Fire Department: 911 (US) / 101 (India)
• Emergency Services: 112
• Building Security: Contact local security

This is an automated alert from the Accurate Fire Detection System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach fire image
            with open(image_path, 'rb') as image_file:
                image_attachment = MIMEImage(image_file.read())
                image_attachment.add_header('Content-Disposition', 'attachment', 
                                          filename=f"accurate_fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                msg.attach(image_attachment)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.login(self.email_config['sender_email'], self.email_config['app_password'])
                server.send_message(msg)
            
            print(f"✅ Accurate fire alert email sent to {self.email_config['recipient_email']}")
            print(f"📷 Fire image saved: {image_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending fire alert email: {e}")
            return False

    def log_fire_event(self, fire_area, confidence):
        """Log accurate fire detection event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'accurate_fire_detected',
            'alert_count': self.alert_count,
            'confidence': confidence,
            'fire_area': fire_area,
            'camera_index': self.camera_index
        }
        
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"system_logs/accurate_fire_events_{timestamp}.json"
        
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

    def run_accurate_detection(self):
        """Run the accurate fire detection loop"""
        print("\n🔥 Starting Accurate Fire Detection System...")
        print("📹 Camera: Initializing...")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return
        
        print("✅ Camera initialized successfully")
        print("🎯 Accurate detection active - Press 'q' to quit")
        print("📧 Email alerts enabled")
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
        
        frame_count = 0
        start_time = time.time()
        prev_frame = None
        
        try:
            while self.system_active:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame!")
                    break
                
                # Process frame for accurate fire detection
                processed_frame, fire_detected, confidence = self.process_frame(frame, prev_frame)
                
                # Display system status
                status_text = f"FIRE DETECTED! ({confidence:.2f})" if fire_detected else "No Fire"
                status_color = (0, 0, 255) if fire_detected else (0, 255, 0)
                
                cv2.putText(processed_frame, f"Status: {status_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(processed_frame, f"Alerts: {self.alert_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"Confidence: {confidence:.2f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Accurate Fire Detection System', processed_frame)
                
                # Update previous frame
                prev_frame = frame.copy()
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Accurate detection stopped by user")
        except Exception as e:
            print(f"❌ Error in accurate detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Accurate fire detection system stopped")

    def show_system_info(self):
        """Show system information"""
        print("\n📊 Accurate Fire Detection System Information")
        print("=" * 50)
        print(f"📧 Email: {self.email_config['sender_email']}")
        print(f"📬 Recipient: {self.email_config['recipient_email']}")
        print(f"📹 Camera: Index {self.camera_index}")
        print(f"🖼️ Resolution: {self.frame_width}x{self.frame_height}")
        print(f"🎯 Confidence Threshold: {self.confidence_threshold}")
        print(f"🚨 Alert Cooldown: {self.alert_cooldown} seconds")
        print(f"📁 Logs Directory: system_logs/")
        print(f"📸 Images Directory: captured_fires/")
        print("\n🔧 Advanced Detection Features:")
        print("• Multiple HSV color ranges")
        print("• Color distribution analysis")
        print("• Shape feature analysis")
        print("• Motion detection")
        print("• Temporal consistency validation")
        print("• Contour complexity analysis")

def main():
    """Main function"""
    print("🔥 Accurate Fire Detection System with Email Alerts")
    print("=" * 60)
    print("This system uses advanced computer vision to:")
    print("✅ Detect fire with high accuracy")
    print("✅ Reduce false positives")
    print("✅ Send real email alerts")
    print("✅ Capture fire images")
    print("✅ Log all events")
    print()
    
    # Create accurate fire detection system
    fire_system = AccurateFireDetection()
    
    if not fire_system.system_active:
        print("❌ System initialization failed!")
        return
    
    # Show system information
    fire_system.show_system_info()
    
    # Start accurate detection
    input("\nPress Enter to start accurate fire detection...")
    fire_system.run_accurate_detection()

if __name__ == "__main__":
    main()
