#!/usr/bin/env python3
"""
Simple script to send a test email notification to your email address
"""

import json
import os
from notification_system import NotificationSystem

def send_test_email():
    """Send a test email notification"""
    print("📧 Forest Fire Detection - Test Email Notification")
    print("=" * 50)
    
    # Get user's email
    recipient_email = input("Enter your email address: ").strip()
    
    if not recipient_email:
        print("❌ Email address is required!")
        return
    
    print(f"\n📧 Sending test notification to: {recipient_email}")
    
    # Create temporary config for Gmail
    gmail_config = {
        "email": {
            "enabled": True,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": input("Enter your Gmail address: ").strip(),
            "sender_password": input("Enter your Gmail app password: ").strip(),
            "recipients": [recipient_email],
            "use_tls": True
        },
        "sms": {
            "enabled": False,
            "twilio_account_sid": "",
            "twilio_auth_token": "",
            "twilio_phone_number": "",
            "recipients": []
        },
        "notification_settings": {
            "fire_confidence_threshold": 0.7,
            "alert_cooldown_minutes": 1,
            "include_image": False,
            "max_image_size_mb": 5.0
        }
    }
    
    # Save config temporarily
    config_file = "temp_email_config.json"
    with open(config_file, 'w') as f:
        json.dump(gmail_config, f, indent=4)
    
    try:
        # Initialize notification system
        print("\n🔧 Initializing notification system...")
        notification_system = NotificationSystem(config_file)
        
        if not notification_system.email_enabled:
            print("❌ Email configuration is invalid!")
            print("Please check your Gmail address and app password.")
            return
        
        # Send test notification
        print("📤 Sending test notification...")
        
        test_data = {
            'bbox': [100, 100, 300, 300],
            'class_id': 1,
            'coordinates': 'Test Location'
        }
        
        success = notification_system.send_fire_alert(
            detection_data=test_data,
            confidence=0.85,
            location="Test Forest Area"
        )
        
        if success:
            print("✅ Test email notification sent successfully!")
            print(f"📧 Check your inbox at: {recipient_email}")
        else:
            print("❌ Failed to send test notification")
            print("Please check your Gmail settings and app password.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have 2-Factor Authentication enabled on your Gmail")
        print("2. Generate an App Password:")
        print("   - Go to Google Account Settings")
        print("   - Security → 2-Step Verification → App passwords")
        print("   - Generate password for 'Mail'")
        print("3. Use the app password (16 characters) as sender_password")
    
    finally:
        # Clean up temporary config file
        if os.path.exists(config_file):
            os.remove(config_file)

def main():
    """Main function"""
    print("🚨 Test Email Notification System")
    print("This will send you a test fire detection alert email!")
    print()
    
    # Show Gmail setup instructions
    print("📋 Before starting, make sure you have:")
    print("1. ✅ Gmail account with 2-Factor Authentication enabled")
    print("2. ✅ App Password generated for 'Mail'")
    print("3. ✅ App Password ready (16 characters)")
    print()
    
    proceed = input("Ready to proceed? (y/n): ").strip().lower()
    
    if proceed in ['y', 'yes']:
        send_test_email()
    else:
        print("👋 Test cancelled. Come back when you're ready!")

if __name__ == "__main__":
    main()
