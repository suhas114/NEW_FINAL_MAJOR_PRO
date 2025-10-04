#!/usr/bin/env python3
"""
Send Real Email Test for G16 Fire Detection System
This script will actually send a test email to your email address
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_real_email():
    """Send a real test email"""
    print("📧 G16 Fire Detection - Real Email Test")
    print("=" * 50)
    
    # Get user's email
    recipient_email = input("Enter your email address to send the test to: ").strip()
    
    if not recipient_email:
        print("❌ Email address is required!")
        return
    
    print(f"\n📧 Sending test email to: {recipient_email}")
    
    # Gmail configuration (you need to provide these)
    print("\n🔧 Gmail Configuration Required:")
    print("To send emails, you need:")
    print("1. A Gmail account")
    print("2. 2-Factor Authentication enabled")
    print("3. An App Password (not your regular password)")
    print()
    
    sender_email = input("Enter your Gmail address: ").strip()
    app_password = input("Enter your Gmail App Password (16 characters): ").strip()
    
    if not all([sender_email, app_password]):
        print("❌ Gmail address and App Password are required!")
        return
    
    # Create email message
    msg = MIMEMultipart()
    msg['Subject'] = "🚨 G16 Fire Detection System - Test Alert 🚨"
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    # Email body
    body = f"""
🚨 G16 FIRE DETECTION SYSTEM TEST ALERT 🚨

This is a test email from the G16 Fire Detection System.

📅 Test Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🔧 System: G16 Fire Detection System
📍 Location: Test Environment
🎯 Purpose: Email functionality verification

✅ If you received this email, the email alert system is working correctly!

📋 System Components:
• ESP32 Development Board
• Flame Sensor (GPIO 18)
• Buzzer Alarm (GPIO 14)
• Red LED Indicator (GPIO 2)
• Green LED Status (GPIO 19)
• Blynk IoT Integration

🔗 Project Files:
• G16_fire_detection.ino - Arduino code
• esp32_python_integration.py - Python integration
• g16_demonstration.py - System demonstration

📞 Contact: {sender_email}

This is an automated test email from the G16 Fire Detection System.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        print("\n📤 Connecting to Gmail...")
        
        # Create secure connection with server
        context = ssl.create_default_context()
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            print("🔐 Logging in to Gmail...")
            server.login(sender_email, app_password)
            
            print("📤 Sending email...")
            server.send_message(msg)
            
            print("✅ Email sent successfully!")
            print(f"📧 Check your inbox at: {recipient_email}")
            
            # Also send to the original email from G16 code
            if recipient_email != "suhas123kichcha@gmail.com":
                print("\n📧 Sending copy to: suhas123kichcha@gmail.com")
                msg['To'] = "suhas123kichcha@gmail.com"
                server.send_message(msg)
                print("✅ Copy sent successfully!")
            
    except smtplib.AuthenticationError:
        print("❌ Authentication failed!")
        print("🔧 Please check:")
        print("   1. Gmail address is correct")
        print("   2. App Password is correct (16 characters)")
        print("   3. 2-Factor Authentication is enabled")
        print("   4. App Password is generated for 'Mail'")
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        print("🔧 Common issues:")
        print("   • Check internet connection")
        print("   • Verify Gmail settings")
        print("   • Ensure App Password is correct")

def show_gmail_setup_help():
    """Show Gmail setup instructions"""
    print("\n📋 Gmail Setup Instructions:")
    print("=" * 40)
    print()
    print("1️⃣ Enable 2-Factor Authentication:")
    print("   • Go to myaccount.google.com")
    print("   • Security → 2-Step Verification → Get started")
    print("   • Follow the setup process")
    print()
    print("2️⃣ Generate App Password:")
    print("   • Go to myaccount.google.com")
    print("   • Security → App passwords")
    print("   • Select 'Mail' as app")
    print("   • Select 'Other' as device")
    print("   • Name it 'G16 Fire Detection'")
    print("   • Click 'Generate'")
    print("   • Copy the 16-character password")
    print()
    print("3️⃣ Use the App Password:")
    print("   • The password looks like: abcd efgh ijkl mnop")
    print("   • Use this password in the script")
    print("   • Do NOT use your regular Gmail password")
    print()

def main():
    """Main function"""
    print("🚨 G16 Fire Detection System - Real Email Test")
    print("This will send you a REAL test email!")
    print()
    
    while True:
        print("📋 Options:")
        print("1. Send real email test")
        print("2. Show Gmail setup instructions")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            send_real_email()
            break
        elif choice == '2':
            show_gmail_setup_help()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-3.")

if __name__ == "__main__":
    main()
