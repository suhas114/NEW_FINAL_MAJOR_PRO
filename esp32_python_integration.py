#!/usr/bin/env python3
"""
ESP32 Fire Detection System - Python Integration
This script provides additional functionality for the G16 fire detection system
"""

import serial
import time
import json
import logging
from datetime import datetime
import requests
import os

class ESP32FireDetection:
    def __init__(self, port='COM3', baudrate=9600):
        """
        Initialize ESP32 Fire Detection System integration
        Args:
            port (str): Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate (int): Serial communication baudrate
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('esp32_fire_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Fire detection state
        self.fire_detected = False
        self.last_fire_time = None
        self.alert_count = 0
        
        # Configuration
        self.config = {
            'alert_cooldown': 30,  # seconds
            'log_alerts': True,
            'save_data': True,
            'notification_email': '9663890904u@gmail.com'
        }

    def connect(self):
        """Connect to ESP32 via serial port"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            self.is_connected = True
            self.logger.info(f"Connected to ESP32 on {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ESP32: {e}")
            return False

    def disconnect(self):
        """Disconnect from ESP32"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            self.logger.info("Disconnected from ESP32")

    def read_serial_data(self):
        """Read data from ESP32 serial port"""
        if not self.is_connected:
            return None
        
        try:
            if self.serial_conn.in_waiting > 0:
                data = self.serial_conn.readline().decode('utf-8').strip()
                return data
        except Exception as e:
            self.logger.error(f"Error reading serial data: {e}")
        return None

    def send_command(self, command):
        """Send command to ESP32"""
        if not self.is_connected:
            return False
        
        try:
            self.serial_conn.write(f"{command}\n".encode('utf-8'))
            self.logger.info(f"Sent command: {command}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return False

    def process_fire_alert(self, data):
        """Process fire detection alerts"""
        if "FIRE DETECTED" in data:
            current_time = time.time()
            
            # Check cooldown period
            if self.last_fire_time and (current_time - self.last_fire_time) < self.config['alert_cooldown']:
                return
            
            self.fire_detected = True
            self.last_fire_time = current_time
            self.alert_count += 1
            
            # Log alert
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'fire_detected',
                'source': 'ESP32',
                'message': data,
                'alert_count': self.alert_count
            }
            
            self.logger.warning(f"🚨 FIRE ALERT: {data}")
            
            # Save alert data
            if self.config['save_data']:
                self.save_alert_data(alert_data)
            
            # Send additional notifications
            self.send_notifications(alert_data)
            
            return alert_data
        
        elif "NO FIRE DETECTED" in data:
            if self.fire_detected:
                self.logger.info("✅ Fire cleared - Area safe")
                self.fire_detected = False
            return None

    def save_alert_data(self, alert_data):
        """Save alert data to file"""
        try:
            filename = f"fire_alerts_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Load existing data
            existing_data = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            
            # Add new alert
            existing_data.append(alert_data)
            
            # Save updated data
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            self.logger.info(f"Alert data saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving alert data: {e}")

    def send_notifications(self, alert_data):
        """Send additional notifications"""
        try:
            # Email notification (using smtplib)
            self.send_email_alert(alert_data)
            
            # Webhook notification (if configured)
            self.send_webhook_alert(alert_data)
            
        except Exception as e:
            self.logger.error(f"Error sending notifications: {e}")

    def send_email_alert(self, alert_data):
        """Send email alert"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Email configuration (update with your settings)
            sender_email = "your-email@gmail.com"
            sender_password = "your-app-password"
            recipient_email = self.config['notification_email']
            
            # Create message
            msg = MIMEMultipart()
            msg['Subject'] = "🚨 ESP32 Fire Detection Alert"
            msg['From'] = sender_email
            msg['To'] = recipient_email
            
            # Email body
            body = f"""
🚨 FIRE DETECTION ALERT 🚨

ESP32 Fire Detection System has detected a fire!

📅 Time: {alert_data['timestamp']}
🔢 Alert Count: {alert_data['alert_count']}
📍 Source: {alert_data['source']}
📝 Message: {alert_data['message']}

🚨 IMMEDIATE ACTION REQUIRED 🚨
- Check the monitored area
- Contact emergency services if needed
- Verify sensor functionality

This is an automated alert from the ESP32 Fire Detection System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent to {recipient_email}")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")

    def send_webhook_alert(self, alert_data):
        """Send webhook alert (if webhook URL is configured)"""
        webhook_url = os.getenv('WEBHOOK_URL', None)
        
        if webhook_url:
            try:
                payload = {
                    'text': f"🚨 Fire Alert from ESP32: {alert_data['message']}",
                    'timestamp': alert_data['timestamp'],
                    'alert_count': alert_data['alert_count']
                }
                
                response = requests.post(webhook_url, json=payload, timeout=5)
                
                if response.status_code == 200:
                    self.logger.info("Webhook alert sent successfully")
                else:
                    self.logger.error(f"Webhook alert failed: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Error sending webhook alert: {e}")

    def get_status(self):
        """Get current system status"""
        return {
            'connected': self.is_connected,
            'fire_detected': self.fire_detected,
            'alert_count': self.alert_count,
            'last_fire_time': self.last_fire_time,
            'port': self.port,
            'baudrate': self.baudrate
        }

    def reset_alerts(self):
        """Reset alert counter"""
        self.alert_count = 0
        self.fire_detected = False
        self.last_fire_time = None
        self.logger.info("Alert counter reset")

    def monitor(self, duration=None):
        """Monitor ESP32 for fire detection alerts"""
        if not self.connect():
            return
        
        self.logger.info("Starting ESP32 fire detection monitoring...")
        start_time = time.time()
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Read serial data
                data = self.read_serial_data()
                
                if data:
                    # Process fire alerts
                    self.process_fire_alert(data)
                    print(f"ESP32: {data}")
                
                time.sleep(0.1)  # Small delay to prevent high CPU usage
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error during monitoring: {e}")
        finally:
            self.disconnect()

def main():
    """Main function for testing"""
    print("🚨 ESP32 Fire Detection System - Python Integration")
    print("=" * 60)
    
    # Create ESP32 interface
    esp32 = ESP32FireDetection(port='COM3')  # Update port as needed
    
    # Show menu
    while True:
        print("\n📋 Menu:")
        print("1. Start monitoring")
        print("2. Check status")
        print("3. Send test command")
        print("4. Reset alerts")
        print("5. View alert logs")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            duration = input("Enter monitoring duration in seconds (or press Enter for continuous): ").strip()
            if duration:
                esp32.monitor(duration=int(duration))
            else:
                esp32.monitor()
        
        elif choice == '2':
            status = esp32.get_status()
            print("\n📊 System Status:")
            for key, value in status.items():
                print(f"   {key}: {value}")
        
        elif choice == '3':
            command = input("Enter command to send to ESP32: ").strip()
            if command:
                esp32.send_command(command)
        
        elif choice == '4':
            esp32.reset_alerts()
        
        elif choice == '5':
            try:
                filename = f"fire_alerts_{datetime.now().strftime('%Y%m%d')}.json"
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        alerts = json.load(f)
                    print(f"\n📋 Today's Alerts ({len(alerts)}):")
                    for alert in alerts[-5:]:  # Show last 5 alerts
                        print(f"   {alert['timestamp']}: {alert['message']}")
                else:
                    print("No alert logs found for today")
            except Exception as e:
                print(f"Error reading logs: {e}")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
