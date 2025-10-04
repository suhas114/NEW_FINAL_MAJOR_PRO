# G16 Fire Detection System - Setup Guide

## 🚨 Overview

The G16 Fire Detection System is an ESP32-based IoT solution that provides real-time fire monitoring with cloud notifications via Blynk. The system uses a flame sensor to detect fire and automatically sends alerts through multiple channels.

## 📋 Components Required

### Hardware Components
- **ESP32 Development Board**
- **Flame Sensor Module** (IR flame detector)
- **Buzzer** (for audible alarm)
- **Red LED** (for fire alert indication)
- **Green LED** (for safe status indication)
- **Breadboard and jumper wires**
- **USB cable** (for programming)

### Software Requirements
- **Arduino IDE**
- **Blynk IoT Platform account**
- **Required Libraries:**
  - BlynkSimpleEsp32
  - WiFi (included with ESP32 board support)

## 🔧 Hardware Setup

### Pin Connections
| Component | ESP32 Pin | Description |
|-----------|-----------|-------------|
| Flame Sensor | GPIO 18 | Fire detection input |
| Buzzer | GPIO 14 | Alarm output |
| Red LED | GPIO 2 | Fire alert indicator |
| Green LED | GPIO 19 | Safe status indicator |

### Wiring Diagram
```
ESP32 Board:
┌─────────────────┐
│                 │
│  GPIO 18 ←── Flame Sensor
│  GPIO 14 ───→ Buzzer
│  GPIO 2  ───→ Red LED
│  GPIO 19 ───→ Green LED
│                 │
└─────────────────┘
```

## 🌐 Blynk Setup

### 1. Create Blynk Account
1. Go to [blynk.io](https://blynk.io)
2. Sign up for a free account
3. Create a new device

### 2. Device Configuration
- **Template ID**: `TMPLkOtFt5y-`
- **Template Name**: `fdc98`
- **Auth Token**: `WyHXrQ1dtwD70iYLRCqTSvcKJ-OYpCC2`
- **Hardware**: ESP32 DevKit
- **Connection Type**: WiFi

### 3. Dashboard Widgets
Add the following widgets to your Blynk dashboard:

1. **LED Widget (V1)**
   - Virtual Pin: V1
   - Label: "Fire Status"
   - Color: Red for fire, Green for safe

2. **Value Display (V0)**
   - Virtual Pin: V0
   - Label: "Fire Detection"
   - Input Pin: V0

3. **Notification Widget**
   - Enable push notifications

## 📱 Mobile App Setup

### 1. Install Blynk IoT App
- Download from Google Play Store or Apple App Store
- Sign in with your Blynk account

### 2. Configure Device
- Scan QR code from Blynk web dashboard
- Or manually enter device credentials

### 3. Set Email Alerts
- Go to Device Settings → Email
- Add recipient email: `9663890904u@gmail.com`
- Configure alert messages

## 🔧 Software Setup

### 1. Arduino IDE Configuration
1. Open Arduino IDE
2. Go to File → Preferences
3. Add ESP32 board manager URL:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to Tools → Board → Boards Manager
5. Install "ESP32" board support

### 2. Install Libraries
1. Go to Tools → Manage Libraries
2. Search and install:
   - **BlynkSimpleEsp32**
   - **WiFi** (should be included)

### 3. Upload Code
1. Open `G16_fire_detection.ino` or `G16_fire_detection_enhanced.ino`
2. Select board: Tools → Board → ESP32 Arduino → ESP32 Dev Module
3. Select port (COM port where ESP32 is connected)
4. Click Upload

## 📡 Network Configuration

### WiFi Settings
Update the WiFi credentials in the code:
```cpp
char ssid[] = "YOUR_WIFI_NAME";
char pass[] = "YOUR_WIFI_PASSWORD";
```

### Network Requirements
- **WiFi Network**: 2.4GHz (ESP32 doesn't support 5GHz)
- **Internet Connection**: Required for Blynk cloud communication
- **Network Security**: WPA2/WPA3 recommended

## 🧪 Testing and Calibration

### 1. Basic Testing
1. Upload code to ESP32
2. Open Serial Monitor (Tools → Serial Monitor)
3. Set baud rate to 9600
4. Check for connection messages:
   ```
   G16 Fire Detection System Starting...
   Connecting to Blynk...
   Connected to Blynk!
   System initialized successfully!
   ```

### 2. Sensor Testing
1. **Safe Status Test**:
   - Green LED should be ON
   - Red LED should be OFF
   - Buzzer should be silent
   - Serial output: "NO FIRE DETECTED - Area safe"

2. **Fire Detection Test**:
   - Bring flame near sensor
   - Red LED should turn ON
   - Green LED should turn OFF
   - Buzzer should sound
   - Serial output: "FIRE DETECTED!!"
   - Check Blynk app for notifications

### 3. Blynk App Testing
1. Open Blynk IoT app
2. Check LED widget status
3. Verify notifications are received
4. Test manual control buttons

## 🔌 Advanced Features

### Enhanced Version Features
- **Alert Cooldown**: Prevents spam notifications (10-second cooldown)
- **Status Tracking**: Maintains fire detection state
- **Manual Control**: Blynk app can trigger alerts manually
- **Better Logging**: Detailed serial output
- **Connection Status**: Real-time connection monitoring

### Customization Options
1. **Sensor Sensitivity**: Adjust flame sensor sensitivity
2. **Alert Timing**: Modify cooldown period
3. **Notification Recipients**: Add more email addresses
4. **LED Colors**: Change indicator LED colors
5. **Buzzer Patterns**: Create custom alarm patterns

## 🛠️ Troubleshooting

### Common Issues

1. **ESP32 Not Connecting to WiFi**
   - Check WiFi credentials
   - Ensure 2.4GHz network
   - Check signal strength

2. **Blynk Connection Failed**
   - Verify auth token
   - Check internet connection
   - Restart ESP32

3. **Sensor Not Responding**
   - Check wiring connections
   - Verify power supply
   - Clean sensor lens

4. **False Alarms**
   - Adjust sensor sensitivity
   - Check for infrared interference
   - Relocate sensor if needed

### Debug Commands
Enable debug output by adding:
```cpp
#define BLYNK_PRINT Serial
#define BLYNK_DEBUG Serial
```

## 📊 Monitoring and Maintenance

### Regular Checks
- **Weekly**: Test sensor functionality
- **Monthly**: Clean sensor lens
- **Quarterly**: Verify Blynk connectivity
- **Annually**: Update firmware if needed

### Performance Metrics
- **Response Time**: <500ms from detection to alert
- **False Alarm Rate**: <1% in normal conditions
- **Uptime**: >99% with stable internet

## 🔒 Security Considerations

### Network Security
- Use strong WiFi passwords
- Enable WPA2/WPA3 encryption
- Regular password updates

### Access Control
- Limit Blynk app access
- Monitor device logs
- Regular security audits

## 📞 Support

### Documentation
- Read this setup guide carefully
- Check Arduino IDE error messages
- Review Blynk documentation

### Community Resources
- Arduino Forum
- ESP32 Community
- Blynk Support

### Contact Information
- Email: `9663890904u@gmail.com`
- Blynk Support: [support.blynk.io](https://support.blynk.io)

---

**⚠️ Important Notes:**
- Keep flame sensor clean for optimal performance
- Test system regularly in different conditions
- Maintain backup power supply for critical applications
- Follow local fire safety regulations
