#!/usr/bin/env python3
"""
G16 Fire Detection System - Comprehensive Demonstration

This script demonstrates the complete G16 Fire Detection System including:
1. Hardware simulation (ESP32 behavior)
2. Blynk IoT integration simulation
3. Python monitoring and integration
4. Real-time alert system
5. Email notifications
6. Data logging and analytics
"""

import time
import json
import threading
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import cv2
import numpy as np

class G16FireDetectionDemo:
    def __init__(self):
        """Initialize the G16 Fire Detection System demonstration"""
        self.fire_detected = False
        self.alert_count = 0
        self.system_active = True
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds
        self.demo_duration = 30  # seconds
        
        # Hardware simulation states
        self.hardware_states = {
            'red_led': False,
            'green_led': True,
            'buzzer': False,
            'flame_sensor': False
        }
        
        # Blynk simulation
        self.blynk_connected = False
        self.notification_count = 0
        
        # Data logging
        self.alert_log = []
        self.event_log = []
        
        print("🚨 G16 Fire Detection System - Demonstration")
        print("=" * 60)

    def simulate_hardware(self):
        """Simulate ESP32 hardware behavior"""
        print("\n🔧 Hardware Simulation Started")
        
        while self.system_active:
            # Simulate flame sensor reading
            if random.random() < 0.1:  # 10% chance of fire detection
                self.detect_fire()
            else:
                self.clear_fire()
            
            time.sleep(2)  # Simulate sensor reading interval

    def detect_fire(self):
        """Simulate fire detection"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.fire_detected = True
        self.last_alert_time = current_time
        self.alert_count += 1
        
        # Update hardware states
        self.hardware_states['red_led'] = True
        self.hardware_states['green_led'] = False
        self.hardware_states['buzzer'] = True
        self.hardware_states['flame_sensor'] = True
        
        # Log event
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'fire_detected',
            'alert_count': self.alert_count,
            'hardware_states': self.hardware_states.copy()
        }
        self.event_log.append(event)
        
        print(f"🚨 FIRE DETECTED! (Alert #{self.alert_count})")
        print("   🔴 Red LED: ON")
        print("   🟢 Green LED: OFF")
        print("   🔊 Buzzer: ACTIVE")
        print("   🔥 Flame Sensor: TRIGGERED")
        
        # Send notifications
        self.send_blynk_notification()
        self.send_email_alert()

    def clear_fire(self):
        """Simulate fire cleared"""
        if self.fire_detected:
            self.fire_detected = False
            
            # Update hardware states
            self.hardware_states['red_led'] = False
            self.hardware_states['green_led'] = True
            self.hardware_states['buzzer'] = False
            self.hardware_states['flame_sensor'] = False
            
            # Log event
            event = {
                'timestamp': datetime.now().isoformat(),
                'event': 'fire_cleared',
                'alert_count': self.alert_count,
                'hardware_states': self.hardware_states.copy()
            }
            self.event_log.append(event)
            
            print("✅ Fire cleared - Area safe")
            print("   🔴 Red LED: OFF")
            print("   🟢 Green LED: ON")
            print("   🔊 Buzzer: SILENT")

    def send_blynk_notification(self):
        """Simulate Blynk IoT notification"""
        if not self.blynk_connected:
            print("📡 Connecting to Blynk...")
            time.sleep(1)
            self.blynk_connected = True
            print("✅ Connected to Blynk IoT platform")
        
        self.notification_count += 1
        
        print(f"📱 Blynk Notification #{self.notification_count}:")
        print("   🚨 PUSH NOTIFICATION: 'Fire detected!'")
        print("   📧 EMAIL ALERT: Sent to suhas123kichcha@gmail.com")
        print("   📊 EVENT LOG: 'fire_alert' logged")

    def send_email_alert(self):
        """Simulate email alert"""
        try:
            # Simulate email sending
            print("📧 EMAIL ALERT DETAILS:")
            print("   📬 To: suhas123kichcha@gmail.com")
            print("   📧 Subject: 🚨 FIRE DETECTION ALERT 🚨")
            print("   📝 Message: Fire detected in monitored area!")
            print("   ⏰ Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Log alert
            alert = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'email',
                'recipient': 'suhas123kichcha@gmail.com',
                'subject': '🚨 FIRE DETECTION ALERT 🚨',
                'message': 'Fire detected in monitored area!',
                'alert_count': self.alert_count
            }
            self.alert_log.append(alert)
            
        except Exception as e:
            print(f"❌ Email alert simulation failed: {e}")

    def create_demo_interface(self):
        """Create a simple demo interface"""
        print("\n🎮 DEMO INTERFACE")
        print("=" * 30)
        print("1. Start Hardware Simulation")
        print("2. View System Status")
        print("3. View Alert History")
        print("4. Send Test Alert")
        print("5. Show Hardware Diagram")
        print("6. Exit")
        
        while True:
            try:
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == '1':
                    self.start_hardware_simulation()
                elif choice == '2':
                    self.show_system_status()
                elif choice == '3':
                    self.show_alert_history()
                elif choice == '4':
                    self.send_test_alert()
                elif choice == '5':
                    self.show_hardware_diagram()
                elif choice == '6':
                    self.system_active = False
                    print("👋 Demo ended. Thank you!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted. Thank you!")
                self.system_active = False
                break

    def start_hardware_simulation(self):
        """Start hardware simulation"""
        print("\n🚀 Starting Hardware Simulation...")
        print("Duration: 30 seconds")
        print("Press Ctrl+C to stop early")
        
        # Start hardware simulation in separate thread
        hardware_thread = threading.Thread(target=self.simulate_hardware)
        hardware_thread.daemon = True
        hardware_thread.start()
        
        # Monitor for specified duration
        start_time = time.time()
        try:
            while time.time() - start_time < self.demo_duration:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Simulation stopped by user")
        
        self.system_active = False

    def show_system_status(self):
        """Show current system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 30)
        print(f"🔄 System Active: {self.system_active}")
        print(f"🔥 Fire Detected: {self.fire_detected}")
        print(f"🔢 Alert Count: {self.alert_count}")
        print(f"📡 Blynk Connected: {self.blynk_connected}")
        print(f"📱 Notifications Sent: {self.notification_count}")
        
        print("\n🔧 Hardware States:")
        for component, state in self.hardware_states.items():
            status = "🟢 ON" if state else "🔴 OFF"
            print(f"   {component.replace('_', ' ').title()}: {status}")

    def show_alert_history(self):
        """Show alert history"""
        print("\n📋 ALERT HISTORY")
        print("=" * 30)
        
        if not self.alert_log:
            print("No alerts recorded yet.")
            return
        
        for i, alert in enumerate(self.alert_log[-5:], 1):  # Show last 5 alerts
            print(f"\n🚨 Alert #{i}")
            print(f"   📅 Time: {alert['timestamp']}")
            print(f"   📧 Type: {alert['alert_type']}")
            print(f"   📬 To: {alert['recipient']}")
            print(f"   📝 Subject: {alert['subject']}")

    def send_test_alert(self):
        """Send a test alert"""
        print("\n🧪 Sending Test Alert...")
        self.detect_fire()
        print("✅ Test alert sent!")

    def show_hardware_diagram(self):
        """Show hardware connection diagram"""
        print("\n🔌 HARDWARE CONNECTION DIAGRAM")
        print("=" * 40)
        print("ESP32 Development Board:")
        print("┌─────────────────────────────┐")
        print("│                             │")
        print("│  GPIO 18 ←── Flame Sensor   │")
        print("│  GPIO 14 ───→ Buzzer        │")
        print("│  GPIO 2  ───→ Red LED       │")
        print("│  GPIO 19 ───→ Green LED     │")
        print("│                             │")
        print("│  WiFi ───→ Internet         │")
        print("│  USB ───→ Computer          │")
        print("└─────────────────────────────┘")

    def save_demo_data(self):
        """Save demonstration data to files"""
        try:
            # Save alert log
            with open('g16_demo_alerts.json', 'w') as f:
                json.dump(self.alert_log, f, indent=2)
            
            # Save event log
            with open('g16_demo_events.json', 'w') as f:
                json.dump(self.event_log, f, indent=2)
            
            print("\n💾 Demo data saved to files:")
            print("   📄 g16_demo_alerts.json")
            print("   📄 g16_demo_events.json")
            
        except Exception as e:
            print(f"❌ Error saving demo data: {e}")

    def run_complete_demo(self):
        """Run the complete demonstration"""
        print("\n🎬 Starting Complete G16 Fire Detection Demo")
        print("This demo will simulate:")
        print("✅ ESP32 hardware behavior")
        print("✅ Flame sensor detection")
        print("✅ LED and buzzer activation")
        print("✅ Blynk IoT notifications")
        print("✅ Email alerts")
        print("✅ Data logging")
        
        input("\nPress Enter to start the demo...")
        
        # Start hardware simulation
        hardware_thread = threading.Thread(target=self.simulate_hardware)
        hardware_thread.daemon = True
        hardware_thread.start()
        
        # Run for demo duration
        start_time = time.time()
        while time.time() - start_time < self.demo_duration:
            time.sleep(1)
        
        self.system_active = False
        
        # Show final summary
        self.show_final_summary()
        
        # Save data
        self.save_demo_data()

    def show_final_summary(self):
        """Show final demonstration summary"""
        print("\n🎯 DEMONSTRATION SUMMARY")
        print("=" * 40)
        print(f"🔥 Total Fire Events: {len([e for e in self.event_log if e['event'] == 'fire_detected'])}")
        print(f"📧 Email Alerts Sent: {len(self.alert_log)}")
        print(f"📱 Blynk Notifications: {self.notification_count}")
        print(f"⏱️ Demo Duration: {self.demo_duration} seconds")
        print(f"📊 Events Logged: {len(self.event_log)}")
        
        print("\n✅ Demo completed successfully!")
        print("The G16 Fire Detection System demonstrates:")
        print("   🚨 Real-time fire detection")
        print("   🔊 Immediate alarm activation")
        print("   📡 Cloud-based notifications")
        print("   📧 Email alert system")
        print("   📊 Comprehensive data logging")

def main():
    """Main function"""
    demo = G16FireDetectionDemo()
    
    print("Welcome to the G16 Fire Detection System Demo!")
    print("Choose your demo mode:")
    print("1. Interactive Demo (Menu-based)")
    print("2. Complete Demo (Automated)")
    
    choice = input("\nSelect demo mode (1 or 2): ").strip()
    
    if choice == '1':
        demo.create_demo_interface()
    elif choice == '2':
        demo.run_complete_demo()
    else:
        print("❌ Invalid choice. Running complete demo...")
        demo.run_complete_demo()

if __name__ == "__main__":
    main()
