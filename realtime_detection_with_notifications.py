
import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np
from collections import deque
import datetime
import pygame
import os
from notification_system import NotificationSystem

class EnhancedDetectionVisualizer:
    def __init__(self, max_points=100, notification_config=None):
        self.fps_history = deque(maxlen=max_points)
        self.detection_history = deque(maxlen=max_points)
        self.max_points = max_points
        self.start_time = time.time()
        
        # UI Control parameters
        self.conf_threshold = 0.25
        self.thermal_alpha = 0.3
        self.show_fps_graph = True
        self.show_info_panel = True
        self.show_thermal = True
        self.show_boxes = True
        self.box_thickness = 2
        self.text_scale = 1.0
        self.thermal_sensitivity = 50
        self.min_temp = 500.0
        self.max_temp = 800.0
        self.alert_temp = 100.0
        self.enable_sound = True
        self.panel_height = 360
        self.panel_width = 300
        
        # Initialize sound alert
        pygame.init()
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert.wav") if self.enable_sound else None
        
        # Initialize notification system
        self.notification_system = None
        if notification_config:
            try:
                self.notification_system = NotificationSystem(notification_config)
                print("✅ Notification system initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize notification system: {e}")
        
        # Alert cooldown system
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes in seconds
        self.fire_detected = False
        
        # Image capture for notifications
        self.capture_images = True
        self.images_dir = "captured_alerts"
        os.makedirs(self.images_dir, exist_ok=True)

    def create_control_window(self):
        cv2.namedWindow('Controls')
        cv2.resizeWindow('Controls', 400, 900)
        
        # Add trackbars
        cv2.createTrackbar('Confidence Threshold', 'Controls', 25, 100, 
                          lambda x: self._update_conf_threshold(x/100))
        cv2.createTrackbar('Thermal Opacity', 'Controls', 30, 100, 
                          lambda x: self._update_thermal_alpha(x/100))
        cv2.createTrackbar('Box Thickness', 'Controls', 2, 10, 
                          lambda x: self._update_box_thickness(x))
        cv2.createTrackbar('Text Scale', 'Controls', 10, 30, 
                          lambda x: self._update_text_scale(x/10))
        cv2.createTrackbar('Thermal Sensitivity', 'Controls', 50, 100,
                          lambda x: self._update_thermal_sensitivity(x))
        cv2.createTrackbar('Alert Temperature', 'Controls', 100, 1000,
                          lambda x: self._update_alert_temp(x))
        
        # Add toggle buttons
        cv2.createTrackbar('Show FPS Graph', 'Controls', 1, 1, 
                          lambda x: self._update_show_fps_graph(bool(x)))
        cv2.createTrackbar('Show Info Panel', 'Controls', 1, 1, 
                          lambda x: self._update_show_info_panel(bool(x)))
        cv2.createTrackbar('Show Thermal', 'Controls', 1, 1, 
                          lambda x: self._update_show_thermal(bool(x)))
        cv2.createTrackbar('Show Boxes', 'Controls', 1, 1, 
                          lambda x: self._update_show_boxes(bool(x)))
        cv2.createTrackbar('Enable Sound', 'Controls', 1, 1,
                          lambda x: self._update_enable_sound(bool(x)))
        cv2.createTrackbar('Capture Images', 'Controls', 1, 1,
                          lambda x: self._update_capture_images(bool(x)))

    # Callback methods for trackbars
    def _update_conf_threshold(self, value): self.conf_threshold = value
    def _update_thermal_alpha(self, value): self.thermal_alpha = value
    def _update_box_thickness(self, value): self.box_thickness = value
    def _update_text_scale(self, value): self.text_scale = value
    def _update_show_fps_graph(self, value): self.show_fps_graph = value
    def _update_show_info_panel(self, value): self.show_info_panel = value
    def _update_show_thermal(self, value): self.show_thermal = value
    def _update_show_boxes(self, value): self.show_boxes = value
    def _update_thermal_sensitivity(self, value): self.thermal_sensitivity = value
    def _update_alert_temp(self, value): self.alert_temp = float(value)
    def _update_enable_sound(self, value): 
        self.enable_sound = value
        if not value and self.alert_sound:
            self.alert_sound.stop()
    def _update_capture_images(self, value): self.capture_images = bool(value)

    def create_info_panel(self, frame_count, fps, detections, temperatures, fire_detections):
        if not self.show_info_panel:
            return None
            
        panel = np.ones((self.panel_height, self.panel_width, 3), dtype=np.uint8) * 255
        
        # Add timestamp and basic stats
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, timestamp, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        
        cv2.putText(panel, f"Frame: {frame_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        cv2.putText(panel, f"Detections: {len(detections)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        
        # Add fire detection status
        if fire_detections:
            cv2.putText(panel, "🔥 FIRE DETECTED! 🔥", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.text_scale, (0, 0, 255), 2)
            
            # Check if notification system is available
            if self.notification_system:
                cv2.putText(panel, "📧 Notifications: Active", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 128, 0), 1)
            else:
                cv2.putText(panel, "📧 Notifications: Disabled", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 255), 1)
        
        # Add temperature information
        if temperatures:
            avg_temp = np.mean(temperatures)
            cv2.putText(panel, f"Avg Temp: {avg_temp:.1f}°C", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
            
            if avg_temp > self.alert_temp:
                cv2.putText(panel, "⚠️ HIGH TEMPERATURE ⚠️", (10, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 255), 1)
        
        # Add notification status
        if self.notification_system:
            if self.notification_system.email_enabled:
                cv2.putText(panel, "📧 Email: Enabled", (10, 230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale, (0, 128, 0), 1)
            if self.notification_system.sms_enabled:
                cv2.putText(panel, "📱 SMS: Enabled", (10, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale, (0, 128, 0), 1)
        
        # Add last alert time
        if self.last_alert_time > 0:
            last_alert_str = datetime.datetime.fromtimestamp(self.last_alert_time).strftime("%H:%M:%S")
            cv2.putText(panel, f"Last Alert: {last_alert_str}", (10, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale, (0, 0, 0), 1)
        
        return panel

    def process_detections(self, results, frame, frame_count):
        """Process YOLO detection results and trigger notifications"""
        current_time = time.time()
        fire_detected_in_frame = False
        highest_confidence = 0.0
        best_detection = None
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                if confidence > self.conf_threshold:
                    # Check if this is a fire-related detection
                    if class_id in [0, 1]:  # fire or smoke classes
                        fire_detected_in_frame = True
                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            best_detection = {
                                'bbox': box.xyxy[0].tolist(),
                                'confidence': confidence,
                                'class_id': class_id,
                                'coordinates': f"Frame {frame_count}"
                            }
        
        # Handle fire detection and notifications
        if fire_detected_in_frame and best_detection:
            self.fire_detected = True
            
            # Play sound alert
            if self.enable_sound and self.alert_sound:
                self.alert_sound.play()
            
            # Send notification if cooldown has passed
            if (current_time - self.last_alert_time) > self.alert_cooldown:
                self._send_fire_alert(best_detection, frame, frame_count)
                self.last_alert_time = current_time
        else:
            self.fire_detected = False
        
        return self.fire_detected, best_detection

    def _send_fire_alert(self, detection_data, frame, frame_count):
        """Send fire alert notification with captured image"""
        if not self.notification_system:
            return
        
        try:
            # Capture and save image
            image_path = None
            if self.capture_images:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"fire_alert_{timestamp}_frame_{frame_count}.jpg"
                image_path = os.path.join(self.images_dir, image_filename)
                cv2.imwrite(image_path, frame)
                print(f"📸 Alert image saved: {image_path}")
            
            # Send notification
            success = self.notification_system.send_fire_alert(
                detection_data=detection_data,
                image_path=image_path,
                confidence=detection_data['confidence'],
                location=f"Frame {frame_count}"
            )
            
            if success:
                print("✅ Fire alert notification sent successfully!")
            else:
                print("❌ Failed to send fire alert notification")
                
        except Exception as e:
            print(f"❌ Error sending fire alert: {e}")

    def run_detection(self, model_path=None, camera_source=0):
        """Run the enhanced real-time detection system"""
        # Load YOLO model
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ Loaded custom model: {model_path}")
        else:
            model = YOLO('yolov8n.pt')
            print("✅ Loaded default YOLOv8 model")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🚀 Starting enhanced real-time fire detection...")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Use the Controls window to adjust parameters")
        print(f"- Notifications: {'Enabled' if self.notification_system else 'Disabled'}")
        
        # Create control window
        self.create_control_window()
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Perform detection
            results = model(frame, conf=self.conf_threshold)
            
            # Process detections and handle notifications
            fire_detected, detection_data = self.process_detections(results, frame, frame_count)
            
            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Update FPS history
            self.fps_history.append(fps)
            
            # Create visualization
            annotated_frame = results[0].plot()
            
            # Create info panel
            info_panel = self.create_info_panel(
                frame_count, fps, 
                results[0].boxes.xyxy if results[0].boxes is not None else [],
                [], fire_detected
            )
            
            # Combine frames
            if info_panel is not None:
                combined_frame = np.hstack([annotated_frame, info_panel])
            else:
                combined_frame = annotated_frame
            
            # Display frame
            cv2.imshow('Enhanced Forest Fire Detection', combined_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                # Send test notification
                if self.notification_system:
                    print("🧪 Sending test notification...")
                    self.notification_system.send_test_notification()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("👋 Detection system stopped")

def main():
    """Main function to run the enhanced detection system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Forest Fire Detection with Notifications")
    parser.add_argument("--model", type=str, help="Path to custom YOLO model")
    parser.add_argument("--camera", type=int, default=0, help="Camera source (default: 0)")
    parser.add_argument("--config", type=str, help="Path to notification configuration file")
    
    args = parser.parse_args()
    
    # Initialize enhanced detection system
    detector = EnhancedDetectionVisualizer(notification_config=args.config)
    
    # Run detection
    detector.run_detection(model_path=args.model, camera_source=args.camera)

if __name__ == "__main__":
    main()
