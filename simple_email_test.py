#!/usr/bin/env python3
"""
Simple Email Test - Works with multiple email providers
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def get_email_provider_config():
    """Get email provider configuration"""
    print("\n📧 Email Provider Setup")
    print("=" * 30)
    print("Choose your email provider:")
    print("1. Gmail (recommended)")
    print("2. Outlook/Hotmail")
    print("3. Yahoo")
    print("4. Custom SMTP")
    
    choice = input("\nSelect provider (1-4): ").strip()
    
    if choice == '1':
        return {
            'server': 'smtp.gmail.com',
            'port': 587,
            'use_tls': True,
            'name': 'Gmail'
        }
    elif choice == '2':
        return {
            'server': 'smtp-mail.outlook.com',
            'port': 587,
            'use_tls': True,
            'name': 'Outlook'
        }
    elif choice == '3':
        return {
            'server': 'smtp.mail.yahoo.com',
            'port': 587,
            'use_tls': True,
            'name': 'Yahoo'
        }
    elif choice == '4':
        server = input("Enter SMTP server: ").strip()
        port = int(input("Enter SMTP port (587): ").strip() or "587")
        return {
            'server': server,
            'port': port,
            'use_tls': True,
            'name': 'Custom'
        }
    else:
        print("❌ Invalid choice. Using Gmail...")
        return {
            'server': 'smtp.gmail.com',
            'port': 587,
            'use_tls': True,
            'name': 'Gmail'
        }

def send_simple_test_email():
    """Send a simple test email"""
    print("📧 Simple Email Test")
    print("=" * 30)
    
    # Get recipient email
    recipient_email = input("Enter email to send to: ").strip()
    
    if not recipient_email:
        print("❌ Email address is required!")
        return
    
    # Get email provider config
    provider = get_email_provider_config()
    
    # Get sender credentials
    print(f"\n🔧 {provider['name']} Configuration:")
    sender_email = input(f"Enter your {provider['name']} email: ").strip()
    
    if provider['name'] == 'Gmail':
        print("🔑 For Gmail, you need an App Password (not regular password)")
        print("   Enable 2-Factor Authentication first, then generate App Password")
        sender_password = input("Enter your Gmail App Password: ").strip()
    else:
        sender_password = input(f"Enter your {provider['name']} password: ").strip()
    
    if not all([sender_email, sender_password]):
        print("❌ Email and password are required!")
        return
    
    # Create email message
    msg = MIMEMultipart()
    msg['Subject'] = "🔥 G16 Fire Detection - Test Email 🔥"
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    # Email body
    body = f"""
🔥 G16 FIRE DETECTION SYSTEM - TEST EMAIL 🔥

Hello!

This is a test email from the G16 Fire Detection System.

📅 Sent at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🔧 System: G16 Fire Detection System
🎯 Purpose: Email functionality test

✅ If you received this email, the system is working!

📋 System Features:
• Real-time fire detection
• Automatic alarms
• Cloud notifications
• Email alerts
• Mobile monitoring

🔗 Project Files:
• G16_fire_detection.ino
• esp32_python_integration.py
• g16_demonstration.py

This is a test email from the G16 Fire Detection System.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        print(f"\n📤 Connecting to {provider['name']}...")
        
        # Create SMTP connection
        if provider['use_tls']:
            server = smtplib.SMTP(provider['server'], provider['port'])
            server.starttls(context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(provider['server'], provider['port'])
        
        print("🔐 Logging in...")
        server.login(sender_email, sender_password)
        
        print("📤 Sending email...")
        server.send_message(msg)
        
        print("✅ Email sent successfully!")
        print(f"📧 Check your inbox at: {recipient_email}")
        
        server.quit()
        
    except smtplib.AuthenticationError:
        print("❌ Authentication failed!")
        if provider['name'] == 'Gmail':
            print("🔧 For Gmail, make sure:")
            print("   1. 2-Factor Authentication is enabled")
            print("   2. You're using an App Password (16 characters)")
            print("   3. Not your regular Gmail password")
        else:
            print("🔧 Check your email and password")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Check your internet connection and email settings")

def main():
    """Main function"""
    print("🔥 G16 Fire Detection - Email Test")
    print("This will send you a real test email!")
    print()
    
    proceed = input("Ready to send test email? (y/n): ").strip().lower()
    
    if proceed in ['y', 'yes']:
        send_simple_test_email()
    else:
        print("👋 Test cancelled.")

if __name__ == "__main__":
    main()
