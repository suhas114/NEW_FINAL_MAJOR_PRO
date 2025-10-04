# 🚀 Quick Start Guide - YOLOv8 Fire Detection

Get your YOLOv8 fire detection system running in minutes!

## ⚡ Quick Start (5 minutes)

### 1. Run the Demo (No Setup Required)
```bash
python yolov8_demo.py
```
This runs the system without email setup - perfect for testing!

### 2. Run Full System (With Email Alerts)
```bash
python yolov8_fire_detection.py
```
Follow the prompts to set up email alerts.

### 3. Test the System
```bash
python test_yolov8_fire_detection.py
```
Verifies everything is working correctly.

## 🎮 Demo Controls

When running the demo, use these keys:
- **'q'** - Quit the system
- **'s'** - Save current frame
- **'t'** - Test with sample images
- **'c'** - Change confidence threshold

## 🔧 Advanced Usage

### Use Specific Model
```bash
python run_yolov8_detection.py --model runs/detect/forest_fire_model6/weights/best.pt
```

### Adjust Sensitivity
```bash
# Higher sensitivity (more detections)
python run_yolov8_detection.py --confidence 0.3

# Lower sensitivity (fewer false positives)
python run_yolov8_detection.py --confidence 0.7
```

### Use Different Camera
```bash
python run_yolov8_detection.py --camera 1
```

### Disable Email Alerts
```bash
python run_yolov8_detection.py --no-email
```

## 📊 What You'll See

- **Real-time camera feed** with detection overlay
- **Bounding boxes** around detected fire/smoke
- **Confidence scores** for each detection
- **Alert counter** showing total detections
- **Saved images** in `captured_fires/` folder

## 🎯 Detection Classes

The system detects:
- **Fire** (red boxes) - Active flames
- **Smoke** (yellow boxes) - Smoke from fires

## ⚠️ Troubleshooting

### Camera Not Working?
```bash
# Try different camera index
python yolov8_demo.py --camera 1
```

### Model Not Loading?
```bash
# Install dependencies
pip install ultralytics opencv-python numpy torch
```

### Low Detection Accuracy?
```bash
# Adjust confidence threshold
python run_yolov8_detection.py --confidence 0.4
```

## 📁 Output Files

- **Detected fires**: `captured_fires/yolov8_fire_*.jpg`
- **Event logs**: `system_logs/yolov8_events_*.json`
- **Test outputs**: `test_output_*.jpg`

## 🎉 You're Ready!

Your YOLOv8 fire detection system is now running! The system will:
- ✅ Detect fire and smoke in real-time
- ✅ Save images when fire is detected
- ✅ Send email alerts (if configured)
- ✅ Log all events for analysis

---

**Need help?** Check the full documentation in `README_YOLOv8_Fire_Detection.md`
