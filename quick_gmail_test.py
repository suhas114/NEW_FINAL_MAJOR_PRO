#!/usr/bin/env python3
"""
Quick Gmail test with App Password instructions
"""

import json
import os
from notification_system import NotificationSystem

def show_gmail_instructions():
    """Show Gmail setup instructions"""
    print("📧 Gmail App Password Setup")
    print("=" * 40)
    print()
    print("To send emails through Gmail, you need an App Password:")
    print()
    print("1️⃣ Enable 2-Factor Authentication:")
    print("   - Go to myaccount.google.com")
    print("   - Security → 2-Step Verification → Get started")
    print()
    print("2️⃣ Generate App Password:")
    print("   - Go to myaccount.google.com")
    print("   - Security → App passwords")
    print("   - Select 'Mail' as app")
    print("   - Select 'Other' as device")
    print("   - Name it 'Forest Fire Detection'")
    print("   - Click 'Generate'")
    print("   - Copy the 16-character password")
    print()
    print("3️⃣ The App Password looks like: abcd efgh ijkl mnop")
    print()

def send_gmail_test():
    """Send Gmail test with App Password"""
    show_gmail_instructions()
    
    print("📧 Gmail Test Email")
    print("=" * 30)
    
    # Get user input
    recipient_email = input("Enter email to send to: ").strip()
    sender_email = input("Enter your Gmail address: ").strip()
    app_password = input("Enter your Gmail App Password (16 chars): ").strip()
    
    if not all([recipient_email, sender_email, app_password]):
        print("❌ All fields are required!")
        return
    
    # Create Gmail config
    gmail_config = {
        "email": {
            "enabled": True,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": sender_email,
            "sender_password": app_password,
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
    
    # Save config
    config_file = "gmail_test_config.json"
    with open(config_file, 'w') as f:
        json.dump(gmail_config, f, indent=4)
    
    try:
        print("\n🔧 Testing Gmail configuration...")
        
        # Initialize notification system
        notification_system = NotificationSystem(config_file)
        
        if not notification_system.email_enabled:
            print("❌ Gmail configuration is invalid!")
            return
        
        # Send test notification
        print("📤 Sending test email...")
        
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
            print("✅ Test email sent successfully!")
            print(f"📧 Check your inbox at: {recipient_email}")
            print("\n📋 The email contains:")
            print("   🚨 FOREST FIRE DETECTION ALERT 🚨")
            print("   📅 Current timestamp")
            print("   🎯 85% confidence fire detection")
            print("   📍 Test Forest Area location")
        else:
            print("❌ Failed to send test email")
            print("\n🔧 Common issues:")
            print("1. App Password is incorrect (should be 16 characters)")
            print("2. 2-Factor Authentication not enabled")
            print("3. App Password not generated for 'Mail' app")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Solution:")
        print("Make sure you're using the App Password (16 chars), not your regular Gmail password!")
    
    finally:
        # Clean up config file
        if os.path.exists(config_file):
            os.remove(config_file)

def main():
    """Main function"""
    print("🚨 Gmail Test for Forest Fire Detection")
    print("This will send you a test notification email!")
    print()
    
    proceed = input("Ready to test Gmail? (y/n): ").strip().lower()
    
    if proceed in ['y', 'yes']:
        send_gmail_test()
    else:
        print("👋 Test cancelled!")
        print("Remember to set up App Password first!")

if __name__ == "__main__":
    main()
