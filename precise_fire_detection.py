#!/usr/bin/env python3
"""
Precise Fire Detection System with Advanced Object Recognition
This system uses sophisticated algorithms to properly distinguish fire from other objects
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

class PreciseFireDetection:
    def __init__(self):
        """Initialize the precise fire detection system"""
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
        
        # Precise detection parameters
        self.high_confidence_threshold = 0.8  # Very high confidence required
        self.min_fire_area = 3000  # Minimum fire area in pixels
        self.max_fire_area = 30000  # Maximum fire area (to avoid false positives)
        self.fire_history = deque(maxlen=15)  # Track fire detection history
        self.background_frames = deque(maxlen=10)  # Background model
        
        # Advanced HSV ranges for precise fire detection
        self.fire_ranges = [
            # Bright orange-red fire
            ((0, 180, 180), (25, 255, 255)),
            # Deep red fire
            ((0, 150, 150), (20, 255, 200)),
            # Yellow-orange flames
            ((15, 150, 150), (35, 255, 255))
        ]
        
        # Create directories
        self.create_directories()
        
        # Load email configuration
        self.load_email_config()
        
        print("🔥 Precise Fire Detection System Initialized")
        print("📧 Email alerts enabled")
        print("🎯 High-precision detection algorithms active")

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

    def build_background_model(self, frame):
        """Build a background model for better object detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.background_frames.append(gray)
        
        if len(self.background_frames) >= 5:
            # Calculate median background
            median_bg = np.median(list(self.background_frames), axis=0).astype(np.uint8)
            return median_bg
        return None

    def detect_foreground_objects(self, frame, background):
        """Detect foreground objects using background subtraction"""
        if background is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Background subtraction
        diff = cv2.absdiff(gray, background)
        
        # Apply threshold to get foreground
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh

    def analyze_fire_characteristics(self, frame, mask):
        """Analyze fire-specific characteristics"""
        if np.sum(mask) == 0:
            return 0, 0, 0
        
        # Get pixels within the mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masked_pixels = hsv[mask > 0]
        
        if len(masked_pixels) == 0:
            return 0, 0, 0
        
        # Analyze color characteristics
        mean_hue = np.mean(masked_pixels[:, 0])
        mean_sat = np.mean(masked_pixels[:, 1])
        mean_val = np.mean(masked_pixels[:, 2])
        
        # Calculate standard deviations
        hue_std = np.std(masked_pixels[:, 0])
        sat_std = np.std(masked_pixels[:, 1])
        val_std = np.std(masked_pixels[:, 2])
        
        # Fire characteristics scoring
        # 1. High saturation and value
        sat_score = mean_sat / 255.0
        val_score = mean_val / 255.0
        
        # 2. Color variation (fire has varying colors)
        variation_score = min((hue_std + sat_std + val_std) / 100.0, 1.0)
        
        return sat_score, val_score, variation_score

    def analyze_edge_characteristics(self, frame, mask):
        """Analyze edge characteristics specific to fire"""
        if np.sum(mask) == 0:
            return 0
        
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density and complexity
        edge_density = np.sum(edges > 0) / np.sum(mask > 0)
        
        # Fire typically has irregular, complex edges
        return edge_density

    def analyze_texture_patterns(self, frame, mask):
        """Analyze texture patterns specific to fire"""
        if np.sum(mask) == 0:
            return 0
        
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gabor filter to detect texture patterns
        kernel = cv2.getGaborKernel((21, 21), 8.1, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # Calculate texture score
        texture_score = np.std(filtered) / 255.0
        
        return texture_score

    def detect_fire_precisely(self, frame, background):
        """Precise fire detection using multiple advanced methods"""
        best_fire_score = 0
        best_contour = None
        best_mask = None
        
        # Build background model
        bg_model = self.build_background_model(frame)
        
        # Detect foreground objects
        foreground = self.detect_foreground_objects(frame, bg_model)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try multiple HSV ranges
        for lower, upper in self.fire_ranges:
            # Create mask for current range
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Combine with foreground detection if available
            if foreground is not None:
                mask = cv2.bitwise_and(mask, foreground)
            
            # Morphological operations
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Check area constraints
                if area < self.min_fire_area or area > self.max_fire_area:
                    continue
                
                # Advanced analysis
                sat_score, val_score, variation_score = self.analyze_fire_characteristics(frame, mask)
                edge_score = self.analyze_edge_characteristics(frame, mask)
                texture_score = self.analyze_texture_patterns(frame, mask)
                
                # Calculate comprehensive fire score
                color_score = (sat_score + val_score + variation_score) / 3.0
                fire_score = (color_score * 0.4 + edge_score * 0.3 + texture_score * 0.3)
                
                if fire_score > best_fire_score:
                    best_fire_score = fire_score
                    best_contour = contour
                    best_mask = mask
        
        return best_fire_score, best_contour, best_mask

    def validate_fire_precisely(self, frame, contour, mask, fire_score):
        """Precise validation to eliminate false positives"""
        if contour is None or fire_score < self.high_confidence_threshold:
            return False, 0
        
        # Additional precise checks
        
        # 1. Check temporal consistency over multiple frames
        self.fire_history.append(fire_score)
        if len(self.fire_history) >= 5:
            recent_scores = list(self.fire_history)[-5:]
            avg_score = np.mean(recent_scores)
            score_variance = np.var(recent_scores)
            
            # Fire should be consistent but with some variation
            if avg_score < self.high_confidence_threshold or score_variance < 0.01:
                return False, fire_score
        
        # 2. Analyze contour properties
        if contour is not None:
            # Calculate contour properties
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if perimeter > 0 and area > 0:
                # Circularity (fire is irregular)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 1
                
                # Solidity (fire has complex shapes)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 1
                
                # Fire validation criteria
                if circularity > 0.8:  # Too circular (likely not fire)
                    return False, fire_score
                
                if solidity > 0.95:  # Too solid (likely not fire)
                    return False, fire_score
        
        # 3. Check color distribution
        if mask is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masked_pixels = hsv[mask > 0]
            
            if len(masked_pixels) > 0:
                # Check for fire-like color distribution
                hue_hist = np.histogram(masked_pixels[:, 0], bins=10, range=(0, 180))[0]
                sat_hist = np.histogram(masked_pixels[:, 1], bins=10, range=(0, 256))[0]
                
                # Fire should have good distribution
                hue_distribution = np.std(hue_hist) / np.mean(hue_hist) if np.mean(hue_hist) > 0 else 0
                sat_distribution = np.std(sat_hist) / np.mean(sat_hist) if np.mean(sat_hist) > 0 else 0
                
                if hue_distribution < 0.3 or sat_distribution < 0.3:
                    return False, fire_score
        
        return True, fire_score

    def process_frame_precisely(self, frame, background):
        """Process frame with precise fire detection"""
        # Detect fire using precise algorithm
        fire_score, contour, mask = self.detect_fire_precisely(frame, background)
        
        # Validate fire detection precisely
        is_valid_fire, final_score = self.validate_fire_precisely(frame, contour, mask, fire_score)
        
        fire_detected = is_valid_fire and final_score >= self.high_confidence_threshold
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
        
        if fire_detected and final_score >= self.high_confidence_threshold:
            if not self.fire_detected and (current_time - self.last_alert_time) > self.alert_cooldown:
                # Fire just detected
                self.fire_detected = True
                self.last_alert_time = current_time
                self.alert_count += 1
                
                print(f"\n🚨 PRECISE FIRE DETECTED! Alert #{self.alert_count}")
                print(f"🎯 Confidence Score: {final_score:.2f}")
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
        filename = f"captured_fires/precise_fire_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

    def send_fire_alert_email(self, frame, fire_area, confidence):
        """Send precise fire alert email"""
        try:
            # Capture fire image
            image_path = self.capture_fire_image(frame)
            
            # Create email message
            msg = MIMEMultipart()
            msg['Subject'] = "🚨 PRECISE FIRE DETECTED - Emergency Alert 🚨"
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            
            # Email body
            body = f"""
🚨 PRECISE FIRE DETECTION ALERT - EMERGENCY 🚨

FIRE CONFIRMED in monitored area with HIGH PRECISION!

📅 Detection Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🎯 Precision: {confidence:.2f} ({confidence*100:.1f}%)
🔥 Fire Area: {fire_area:.0f} pixels
🔢 Alert Count: {self.alert_count}
📍 Location: High-Precision Camera Monitoring System

⚠️ IMMEDIATE ACTION REQUIRED:
• Check the monitored area immediately
• Contact emergency services if needed
• Verify fire sensor functionality
• Review security camera footage

🔧 High-Precision System Information:
• Detection Method: Advanced Computer Vision
• Precision: {confidence*100:.1f}% confidence
• Validation: Multi-stage verification
• Background: Adaptive background modeling
• Status: FIRE CONFIRMED WITH HIGH PRECISION

📞 Emergency Contacts:
• Fire Department: 911 (US) / 101 (India)
• Emergency Services: 112
• Building Security: Contact local security

This is an automated alert from the Precise Fire Detection System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach fire image
            with open(image_path, 'rb') as image_file:
                image_attachment = MIMEImage(image_file.read())
                image_attachment.add_header('Content-Disposition', 'attachment', 
                                          filename=f"precise_fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                msg.attach(image_attachment)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.login(self.email_config['sender_email'], self.email_config['app_password'])
                server.send_message(msg)
            
            print(f"✅ Precise fire alert email sent to {self.email_config['recipient_email']}")
            print(f"📷 Fire image saved: {image_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending fire alert email: {e}")
            return False

    def log_fire_event(self, fire_area, confidence):
        """Log precise fire detection event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'precise_fire_detected',
            'alert_count': self.alert_count,
            'confidence': confidence,
            'fire_area': fire_area,
            'camera_index': self.camera_index
        }
        
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"system_logs/precise_fire_events_{timestamp}.json"
        
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

    def run_precise_detection(self):
        """Run the precise fire detection loop"""
        print("\n🔥 Starting Precise Fire Detection System...")
        print("📹 Camera: Initializing...")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return
        
        print("✅ Camera initialized successfully")
        print("🎯 Precise detection active - Press 'q' to quit")
        print("📧 Email alerts enabled")
        print(f"🎯 High confidence threshold: {self.high_confidence_threshold}")
        
        frame_count = 0
        start_time = time.time()
        background = None
        
        try:
            while self.system_active:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame!")
                    break
                
                # Process frame for precise fire detection
                processed_frame, fire_detected, confidence = self.process_frame_precisely(frame, background)
                
                # Update background model
                background = self.build_background_model(frame)
                
                # Display system status
                status_text = f"PRECISE FIRE! ({confidence:.2f})" if fire_detected else "No Fire"
                status_color = (0, 0, 255) if fire_detected else (0, 255, 0)
                
                cv2.putText(processed_frame, f"Status: {status_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(processed_frame, f"Alerts: {self.alert_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"Precision: {confidence:.2f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Precise Fire Detection System', processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Precise detection stopped by user")
        except Exception as e:
            print(f"❌ Error in precise detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Precise fire detection system stopped")

    def show_system_info(self):
        """Show system information"""
        print("\n📊 Precise Fire Detection System Information")
        print("=" * 55)
        print(f"📧 Email: {self.email_config['sender_email']}")
        print(f"📬 Recipient: {self.email_config['recipient_email']}")
        print(f"📹 Camera: Index {self.camera_index}")
        print(f"🖼️ Resolution: {self.frame_width}x{self.frame_height}")
        print(f"🎯 High Confidence Threshold: {self.high_confidence_threshold}")
        print(f"🚨 Alert Cooldown: {self.alert_cooldown} seconds")
        print(f"📁 Logs Directory: system_logs/")
        print(f"📸 Images Directory: captured_fires/")
        print("\n🔧 Precise Detection Features:")
        print("• Background subtraction and modeling")
        print("• Advanced HSV color analysis")
        print("• Edge characteristic analysis")
        print("• Texture pattern recognition")
        print("• Multi-stage validation")
        print("• Temporal consistency checking")
        print("• Contour property analysis")

def main():
    """Main function"""
    print("🔥 Precise Fire Detection System with Email Alerts")
    print("=" * 60)
    print("This system uses highly precise computer vision to:")
    print("✅ Detect fire with maximum accuracy")
    print("✅ Eliminate false positives")
    print("✅ Send real email alerts")
    print("✅ Capture fire images")
    print("✅ Log all events")
    print()
    
    # Create precise fire detection system
    fire_system = PreciseFireDetection()
    
    if not fire_system.system_active:
        print("❌ System initialization failed!")
        return
    
    # Show system information
    fire_system.show_system_info()
    
    # Start precise detection
    input("\nPress Enter to start precise fire detection...")
    fire_system.run_precise_detection()

if __name__ == "__main__":
    main()
