#!/usr/bin/env python3
"""
G16 Fire Detection System - Visual Demonstration

This script creates a visual demonstration of the G16 Fire Detection System
using OpenCV to show the hardware components, status indicators, and alerts.
"""

import cv2
import numpy as np
import time
import json
import threading
import random
from datetime import datetime

class G16VisualDemo:
    def __init__(self):
        """Initialize the visual demonstration"""
        self.fire_detected = False
        self.alert_count = 0
        self.system_active = True
        self.last_alert_time = 0
        self.alert_cooldown = 5
        
        # Hardware states
        self.hardware_states = {
            'red_led': False,
            'green_led': True,
            'buzzer': False,
            'flame_sensor': False
        }
        
        # Demo events
        self.events = []
        
        # Window settings
        self.window_width = 1200
        self.window_height = 800
        self.panel_width = 300
        
        print("🎬 G16 Fire Detection System - Visual Demonstration")

    def create_main_window(self):
        """Create the main demonstration window"""
        # Create window
        cv2.namedWindow('G16 Fire Detection System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('G16 Fire Detection System', self.window_width, self.window_height)
        
        while self.system_active:
            # Create main frame
            frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            
            # Draw ESP32 board
            self.draw_esp32_board(frame)
            
            # Draw status panel
            self.draw_status_panel(frame)
            
            # Draw alert history
            self.draw_alert_history(frame)
            
            # Draw system diagram
            self.draw_system_diagram(frame)
            
            # Show frame
            cv2.imshow('G16 Fire Detection System', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.trigger_test_alert()
        
        cv2.destroyAllWindows()

    def draw_esp32_board(self, frame):
        """Draw ESP32 development board"""
        # Board outline
        board_x = 50
        board_y = 100
        board_w = 400
        board_h = 300
        
        # Draw board
        cv2.rectangle(frame, (board_x, board_y), (board_x + board_w, board_y + board_h), (100, 100, 100), 2)
        cv2.putText(frame, 'ESP32 Development Board', (board_x, board_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw GPIO pins
        gpio_pins = [
            {'name': 'GPIO 18', 'x': board_x + 30, 'y': board_y + 50, 'connected': 'Flame Sensor'},
            {'name': 'GPIO 14', 'x': board_x + 30, 'y': board_y + 100, 'connected': 'Buzzer'},
            {'name': 'GPIO 2', 'x': board_x + 30, 'y': board_y + 150, 'connected': 'Red LED'},
            {'name': 'GPIO 19', 'x': board_x + 30, 'y': board_y + 200, 'connected': 'Green LED'}
        ]
        
        for pin in gpio_pins:
            # Pin color based on state
            if pin['connected'] == 'Flame Sensor':
                color = (0, 255, 255) if self.hardware_states['flame_sensor'] else (100, 100, 100)
            elif pin['connected'] == 'Buzzer':
                color = (0, 165, 255) if self.hardware_states['buzzer'] else (100, 100, 100)
            elif pin['connected'] == 'Red LED':
                color = (0, 0, 255) if self.hardware_states['red_led'] else (100, 100, 100)
            elif pin['connected'] == 'Green LED':
                color = (0, 255, 0) if self.hardware_states['green_led'] else (100, 100, 100)
            
            # Draw pin
            cv2.circle(frame, (pin['x'], pin['y']), 8, color, -1)
            cv2.putText(frame, pin['name'], (pin['x'] + 15, pin['y'] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, pin['connected'], (pin['x'] + 15, pin['y'] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def draw_status_panel(self, frame):
        """Draw status panel"""
        panel_x = self.window_width - self.panel_width - 20
        panel_y = 20
        
        # Panel background
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + 400), (50, 50, 50), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + 400), (200, 200, 200), 2)
        
        # Title
        cv2.putText(frame, 'System Status', (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status indicators
        status_y = panel_y + 60
        line_height = 30
        
        # Fire detection status
        fire_color = (0, 0, 255) if self.fire_detected else (0, 255, 0)
        fire_text = "FIRE DETECTED!" if self.fire_detected else "No Fire"
        cv2.putText(frame, f'Fire Status: {fire_text}', (panel_x + 10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fire_color, 2)
        status_y += line_height
        
        # Alert count
        cv2.putText(frame, f'Alerts: {self.alert_count}', (panel_x + 10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += line_height
        
        # Hardware states
        cv2.putText(frame, 'Hardware States:', (panel_x + 10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += line_height
        
        for component, state in self.hardware_states.items():
            color = (0, 255, 0) if state else (100, 100, 100)
            status = "ON" if state else "OFF"
            cv2.putText(frame, f'{component}: {status}', (panel_x + 20, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            status_y += 20
        
        # Controls
        status_y += 20
        cv2.putText(frame, 'Controls:', (panel_x + 10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += line_height
        cv2.putText(frame, 'T - Test Alert', (panel_x + 10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        status_y += 20
        cv2.putText(frame, 'Q - Quit', (panel_x + 10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def draw_alert_history(self, frame):
        """Draw alert history"""
        history_x = 50
        history_y = 450
        history_w = 400
        history_h = 200
        
        # History panel
        cv2.rectangle(frame, (history_x, history_y), (history_x + history_w, history_y + history_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (history_x, history_y), (history_x + history_w, history_y + history_h), (200, 200, 200), 2)
        
        # Title
        cv2.putText(frame, 'Alert History', (history_x + 10, history_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Recent events
        event_y = history_y + 50
        recent_events = self.events[-5:]  # Last 5 events
        
        for event in recent_events:
            color = (0, 0, 255) if event['type'] == 'fire_detected' else (0, 255, 0)
            cv2.putText(frame, f"{event['time']} - {event['type']}", (history_x + 10, event_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            event_y += 20
        
        if not recent_events:
            cv2.putText(frame, 'No events yet', (history_x + 10, event_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def draw_system_diagram(self, frame):
        """Draw system diagram"""
        diagram_x = self.window_width - self.panel_width - 420
        diagram_y = 450
        diagram_w = 400
        diagram_h = 200
        
        # Diagram panel
        cv2.rectangle(frame, (diagram_x, diagram_y), (diagram_x + diagram_w, diagram_y + diagram_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (diagram_x, diagram_y), (diagram_x + diagram_w, diagram_y + diagram_h), (200, 200, 200), 2)
        
        # Title
        cv2.putText(frame, 'System Flow', (diagram_x + 10, diagram_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Flow diagram
        flow_y = diagram_y + 50
        
        # Flame Sensor
        cv2.rectangle(frame, (diagram_x + 20, flow_y), (diagram_x + 120, flow_y + 40), (0, 255, 255), 2)
        cv2.putText(frame, 'Flame Sensor', (diagram_x + 30, flow_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Arrow
        cv2.arrowedLine(frame, (diagram_x + 120, flow_y + 20), (diagram_x + 180, flow_y + 20), (255, 255, 255), 2)
        
        # ESP32
        cv2.rectangle(frame, (diagram_x + 180, flow_y), (diagram_x + 280, flow_y + 40), (100, 100, 100), 2)
        cv2.putText(frame, 'ESP32', (diagram_x + 200, flow_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Second row
        flow_y += 60
        
        # Blynk
        cv2.rectangle(frame, (diagram_x + 180, flow_y), (diagram_x + 280, flow_y + 40), (0, 165, 255), 2)
        cv2.putText(frame, 'Blynk IoT', (diagram_x + 190, flow_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Arrow down
        cv2.arrowedLine(frame, (diagram_x + 230, flow_y), (diagram_x + 230, flow_y + 30), (255, 255, 255), 2)
        
        # Email
        cv2.rectangle(frame, (diagram_x + 180, flow_y + 40), (diagram_x + 280, flow_y + 80), (0, 255, 0), 2)
        cv2.putText(frame, 'Email Alert', (diagram_x + 190, flow_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def trigger_test_alert(self):
        """Trigger a test alert"""
        if time.time() - self.last_alert_time > self.alert_cooldown:
            self.detect_fire()

    def detect_fire(self):
        """Simulate fire detection"""
        current_time = time.time()
        
        self.fire_detected = True
        self.last_alert_time = current_time
        self.alert_count += 1
        
        # Update hardware states
        self.hardware_states['red_led'] = True
        self.hardware_states['green_led'] = False
        self.hardware_states['buzzer'] = True
        self.hardware_states['flame_sensor'] = True
        
        # Add event
        self.events.append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'type': 'fire_detected',
            'alert_count': self.alert_count
        })
        
        print(f"🚨 FIRE DETECTED! Alert #{self.alert_count}")

    def clear_fire(self):
        """Simulate fire cleared"""
        if self.fire_detected:
            self.fire_detected = False
            
            # Update hardware states
            self.hardware_states['red_led'] = False
            self.hardware_states['green_led'] = True
            self.hardware_states['buzzer'] = False
            self.hardware_states['flame_sensor'] = False
            
            # Add event
            self.events.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'type': 'fire_cleared',
                'alert_count': self.alert_count
            })
            
            print("✅ Fire cleared - Area safe")

    def simulate_hardware(self):
        """Simulate hardware behavior"""
        while self.system_active:
            # Random fire detection
            if random.random() < 0.05:  # 5% chance
                self.detect_fire()
            elif self.fire_detected and random.random() < 0.1:  # 10% chance to clear
                self.clear_fire()
            
            time.sleep(3)

    def run_demo(self):
        """Run the visual demonstration"""
        print("Starting visual demonstration...")
        print("Controls:")
        print("  T - Trigger test alert")
        print("  Q - Quit")
        
        # Start hardware simulation in background
        hardware_thread = threading.Thread(target=self.simulate_hardware)
        hardware_thread.daemon = True
        hardware_thread.start()
        
        # Start visual interface
        self.create_main_window()
        
        self.system_active = False
        print("Demo ended.")

def main():
    """Main function"""
    demo = G16VisualDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
