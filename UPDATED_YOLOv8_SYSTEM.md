# 🔥 Updated YOLOv8 Fire & Object Detection System

## ✅ Updates Made

### 1. Email Configuration
- **Pre-configured email credentials**:
  - Sender: `suhas123kichcha@gmail.com`
  - App Password: `baka yybc wcbj kawv`
  - Recipient: `suhas123kichcha@gmail.com`
- **No more manual setup required** - system automatically uses your credentials
- **Email alerts enabled by default**

### 2. Object Detection Added
- **General object detection** using YOLOv8 model
- **Detectable objects**: person, car, truck, bus, motorcycle, bicycle, dog, cat
- **Blue bounding boxes** for objects (red for fire, yellow for smoke)
- **Configurable object classes** and minimum area thresholds

### 3. Enhanced Features

#### Visual Improvements
- **Color-coded detections**:
  - 🔴 Red boxes: Fire
  - 🟡 Yellow boxes: Smoke  
  - 🔵 Blue boxes: Objects
- **Real-time object counter** displayed on screen
- **Enhanced status overlay** showing both fire and object detections

#### Email Alerts Enhanced
- **Separate sections** for fire and object detections
- **Detailed detection information** in email body
- **Annotated images** with all detections included

#### Logging Improvements
- **Separate logging** for fire and object detections
- **Enhanced JSON structure** with detection types
- **Comprehensive event tracking**

## 🚀 How to Use

### Quick Start
```bash
# Run with all features enabled
python yolov8_fire_detection.py

# Run demo (no email setup needed)
python yolov8_demo.py

# Test the system
python test_yolov8_fire_detection.py
```

### Command Line Options
```bash
# Adjust sensitivity
python run_yolov8_detection.py --confidence 0.7

# Use different camera
python run_yolov8_detection.py --camera 1

# Disable email alerts
python run_yolov8_detection.py --no-email
```

## 📊 Detection Capabilities

### Fire Detection
- ✅ Fire flames detection
- ✅ Smoke detection
- ✅ High confidence thresholds
- ✅ Stability analysis to reduce false positives

### Object Detection
- ✅ Person detection
- ✅ Vehicle detection (car, truck, bus, motorcycle)
- ✅ Animal detection (dog, cat)
- ✅ Bicycle detection
- ✅ Configurable object classes

## 🎮 Controls

### During Detection
- **'q'** - Quit system
- **'s'** - Save current frame
- **'t'** - Test with sample images (demo only)
- **'c'** - Change confidence threshold (demo only)

## 📧 Email Alerts

When fire is detected, you'll receive an email with:
- 🔥 **Fire detection details** (class, confidence)
- 📦 **Object detection details** (if any objects present)
- 📸 **Annotated image** showing all detections
- ⏰ **Timestamp** and location information

## 📁 Output Files

- **Fire images**: `captured_fires/yolov8_fire_*.jpg`
- **Event logs**: `system_logs/yolov8_events_*.json`
- **Test outputs**: `test_output_*.jpg`

## 🔧 Configuration

### Object Detection Settings
```python
# Enable/disable object detection
self.enable_object_detection = True

# Object classes to detect
self.object_detection_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'dog', 'cat']

# Minimum object area
self.min_object_area = 500
```

### Detection Thresholds
```python
# Fire detection
self.confidence_threshold = 0.5
self.min_fire_area = 1000
self.max_fire_area = 100000

# Object detection
self.min_object_area = 500
```

## 🎯 Performance Tips

- **Higher confidence** (0.7-0.8) for fewer false positives
- **Lower confidence** (0.3-0.4) for higher sensitivity
- **Good lighting** improves detection accuracy
- **Stable camera** reduces false detections

## ⚠️ Safety Notice

This system is for detection purposes only. Always have proper fire safety measures and emergency procedures in place.

---

**System Status**: ✅ Ready to use with email alerts and object detection enabled!
