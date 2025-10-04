#!/usr/bin/env python3
"""
Demonstration script for the Forest Fire Detection Notification System
This script shows how the notification system works without requiring real credentials
"""

import os
import json
from notification_system import NotificationSystem

def create_demo_config():
    """Create a demo configuration for testing"""
    demo_config = {
        "email": {
            "enabled": False,  # Set to True when you have real credentials
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "demo@example.com",
            "sender_password": "demo-password",
            "recipients": ["demo@example.com"],
            "use_tls": True
        },
        "sms": {
            "enabled": False,  # Set to True when you have Twilio credentials
            "twilio_account_sid": "demo-sid",
            "twilio_auth_token": "demo-token",
            "twilio_phone_number": "+1234567890",
            "recipients": ["+1234567890"]
        },
        "notification_settings": {
            "fire_confidence_threshold": 0.7,
            "alert_cooldown_minutes": 1,  # Short cooldown for demo
            "include_image": True,
            "max_image_size_mb": 5.0
        }
    }
    
    # Save demo config
    with open('demo_config.json', 'w') as f:
        json.dump(demo_config, f, indent=4)
    
    print("✅ Demo configuration created: demo_config.json")
    return 'demo_config.json'

def demonstrate_notification_system():
    """Demonstrate the notification system functionality"""
    print("🔥 Forest Fire Detection Notification System Demo")
    print("=" * 60)
    
    # Create demo configuration
    config_file = create_demo_config()
    
    # Initialize notification system
    print("\n1️⃣ Initializing notification system...")
    notification_system = NotificationSystem(config_file)
    
    print("\n2️⃣ System Status:")
    print(f"   📧 Email notifications: {'Enabled' if notification_system.email_enabled else 'Disabled'}")
    print(f"   📱 SMS notifications: {'Enabled' if notification_system.sms_enabled else 'Disabled'}")
    
    # Simulate fire detection
    print("\n3️⃣ Simulating fire detection...")
    test_detection_data = {
        'bbox': [150, 200, 350, 400],
        'confidence': 0.85,
        'class_id': 1,  # Fire class
        'coordinates': 'Demo Location - GPS: 40.7128, -74.0060'
    }
    
    print(f"   🔍 Detection Details:")
    print(f"      - Confidence: {test_detection_data['confidence']:.2%}")
    print(f"      - Bounding Box: {test_detection_data['bbox']}")
    print(f"      - Location: {test_detection_data['coordinates']}")
    
    # Show what the alert message would look like
    print("\n4️⃣ Alert Message Preview:")
    message = notification_system._create_alert_message(
        test_detection_data, 
        test_detection_data['confidence'], 
        "Demo Forest Area"
    )
    print(message)
    
    # Demonstrate notification sending (will fail gracefully since credentials are demo)
    print("\n5️⃣ Attempting to send notifications...")
    success = notification_system.send_fire_alert(
        detection_data=test_detection_data,
        image_path=None,  # No image for demo
        confidence=test_detection_data['confidence'],
        location="Demo Forest Area - GPS: 40.7128, -74.0060"
    )
    
    if success:
        print("   ✅ Notifications sent successfully!")
    else:
        print("   ❌ Notifications failed (expected with demo credentials)")
    
    print("\n6️⃣ Integration Example:")
    print("   To integrate with your existing detection code:")
    print("   ```python")
    print("   from notification_system import NotificationSystem")
    print("   ")
    print("   # Initialize notification system")
    print("   notification_system = NotificationSystem('demo_config.json')")
    print("   ")
    print("   # In your detection loop, when fire is detected:")
    print("   if fire_detected and confidence > threshold:")
    print("       notification_system.send_fire_alert(")
    print("           detection_data=detection_data,")
    print("           image_path=captured_image_path,")
    print("           confidence=confidence,")
    print("           location='Your Location'")
    print("       )")
    print("   ```")
    
    print("\n🎯 Next Steps:")
    print("   1. Update demo_config.json with your real credentials")
    print("   2. Set email.enabled or sms.enabled to True")
    print("   3. Run: python test_notifications.py --config demo_config.json --test-email")
    print("   4. Integrate with your detection system")

def show_configuration_guide():
    """Show configuration guide"""
    print("\n📋 Configuration Guide")
    print("=" * 40)
    
    print("\n📧 Email Configuration (Gmail):")
    print("1. Enable 2-Factor Authentication on your Gmail account")
    print("2. Generate App Password:")
    print("   - Go to Google Account Settings")
    print("   - Security → 2-Step Verification → App passwords")
    print("   - Generate password for 'Mail'")
    print("3. Update demo_config.json:")
    print("   - Set email.enabled to True")
    print("   - Set sender_email to your Gmail")
    print("   - Set sender_password to your app password")
    print("   - Set recipients to target email addresses")
    
    print("\n📱 SMS Configuration (Twilio):")
    print("1. Sign up at twilio.com")
    print("2. Get your credentials from Twilio Console")
    print("3. Update demo_config.json:")
    print("   - Set sms.enabled to True")
    print("   - Add your Twilio credentials")
    print("   - Set recipients to target phone numbers")
    
    print("\n🔧 Test Your Configuration:")
    print("python test_notifications.py --config demo_config.json --test-email")
    print("python test_notifications.py --config demo_config.json --test-sms")

def main():
    """Main demonstration function"""
    print("🚀 Welcome to the Forest Fire Detection Notification System!")
    
    while True:
        print("\n" + "="*60)
        print("📋 Demo Menu:")
        print("1. Run full demonstration")
        print("2. Show configuration guide")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            demonstrate_notification_system()
        elif choice == '2':
            show_configuration_guide()
        elif choice == '3':
            print("👋 Thank you for trying the notification system!")
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
