# YOLOv8 Fire Detection System

A comprehensive fire detection system using YOLOv8 (You Only Look Once version 8) for real-time fire and smoke detection with email alerts and logging capabilities.

## 🚀 Features

- **Real-time Fire Detection**: Uses YOLOv8 for accurate fire and smoke detection
- **Trained Model Support**: Automatically uses your trained fire detection models
- **Email Alerts**: Sends immediate email notifications when fire is detected
- **Image Capture**: Saves detected fire images with bounding boxes
- **Event Logging**: Logs all detection events to JSON files
- **Stability Analysis**: Reduces false positives with detection stability analysis
- **Configurable Parameters**: Adjustable confidence thresholds and detection parameters

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Webcam or camera device
- Internet connection (for email alerts)

### Python Dependencies
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
```

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python test_yolov8_fire_detection.py
   ```

## 🎯 Usage

### Quick Start

Run the fire detection system with default settings:
```bash
python yolov8_fire_detection.py
```

### Using the Runner Script

The `run_yolov8_detection.py` script provides additional options:

```bash
# Basic usage
python run_yolov8_detection.py

# Use specific model
python run_yolov8_detection.py --model runs/detect/forest_fire_model6/weights/best.pt

# Adjust confidence threshold
python run_yolov8_detection.py --confidence 0.7

# Use different camera
python run_yolov8_detection.py --camera 1

# Disable email alerts
python run_yolov8_detection.py --no-email

# Run test mode
python run_yolov8_detection.py --test
```

### Command Line Options

- `--model`: Path to YOLOv8 model file (.pt)
- `--camera`: Camera index (default: 0)
- `--confidence`: Confidence threshold (default: 0.5)
- `--test`: Run in test mode
- `--no-email`: Disable email alerts

## 🔧 Configuration

### Model Selection

The system automatically selects the best available model in this order:
1. `runs/detect/forest_fire_model6/weights/best.pt` (your trained model)
2. `runs/detect/forest_fire_model2/weights/best.pt` (alternative trained model)
3. `yolov8n.pt` (base YOLOv8 model)

### Detection Parameters

You can adjust these parameters in the code:

```python
# Detection thresholds
confidence_threshold = 0.5    # Minimum confidence for detection
iou_threshold = 0.45         # Non-maximum suppression threshold
min_fire_area = 1000         # Minimum fire area in pixels
max_fire_area = 100000       # Maximum fire area

# Alert settings
alert_cooldown = 30          # Seconds between alerts
```

### Email Configuration

To enable email alerts:

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate App Password**:
   - Go to myaccount.google.com
   - Security → 2-Step Verification → App passwords
   - Select "Mail" and generate password
3. **Enter credentials** when prompted by the system

## 📁 File Structure

```
├── yolov8_fire_detection.py          # Main detection system
├── test_yolov8_fire_detection.py     # Test script
├── run_yolov8_detection.py           # Runner script
├── requirements.txt                   # Python dependencies
├── yolov8n.pt                        # Base YOLOv8 model
├── runs/detect/                      # Trained models
│   ├── forest_fire_model6/
│   │   └── weights/best.pt           # Best trained model
│   └── forest_fire_model2/
│       └── weights/best.pt           # Alternative model
├── captured_fires/                   # Saved fire images
├── system_logs/                      # Event logs
└── dataset/                          # Training dataset
```

## 🧪 Testing

### Test the System

Run the test suite to verify everything works:
```bash
python test_yolov8_fire_detection.py
```

### Test with Sample Images

The test script will automatically test with available fire images in:
- `captured_fires/`
- `demo_fires/`

### Test Camera Feed

The test includes a 5-second camera feed test to verify camera functionality.

## 📊 Detection Classes

The system detects two main classes:
- **Fire** (class 0): Active flames and fire
- **Smoke** (class 1): Smoke from fires

## 🔍 How It Works

1. **Frame Capture**: Continuously captures frames from the camera
2. **YOLOv8 Inference**: Runs YOLOv8 model on each frame
3. **Detection Filtering**: Filters detections based on confidence and area
4. **Stability Analysis**: Analyzes detection stability to reduce false positives
5. **Alert Generation**: Triggers alerts when stable fire is detected
6. **Image Saving**: Saves annotated images with detection boxes
7. **Email Notification**: Sends email alerts with attached images
8. **Event Logging**: Logs all events to JSON files

## 📧 Email Alerts

When fire is detected, the system sends an email with:
- Detection timestamp
- Detection details (class, confidence)
- Annotated image showing the fire
- Immediate action instructions

## 📝 Logging

All detection events are logged to JSON files in `system_logs/`:
- Timestamp of detection
- Detection details
- Image path
- Model used
- Confidence threshold

## 🎮 Controls

During detection:
- **'q'**: Quit the detection system
- **'s'**: Save current frame manually

## 🔧 Troubleshooting

### Common Issues

1. **Camera not found**:
   - Check camera index: `--camera 1` or `--camera 2`
   - Verify camera is not in use by another application

2. **Model loading error**:
   - Ensure ultralytics is installed: `pip install ultralytics`
   - Check model file path

3. **Email not sending**:
   - Verify Gmail app password is correct
   - Check 2-factor authentication is enabled
   - Ensure internet connection

4. **Low detection accuracy**:
   - Adjust confidence threshold: `--confidence 0.7`
   - Check lighting conditions
   - Verify camera positioning

### Performance Tips

- **Higher confidence threshold** (0.7-0.8) for fewer false positives
- **Lower confidence threshold** (0.3-0.4) for higher sensitivity
- **Good lighting** improves detection accuracy
- **Stable camera** reduces false detections

## 🤝 Contributing

To improve the system:

1. **Train better models** with more diverse fire/smoke data
2. **Adjust detection parameters** for your specific environment
3. **Add additional alert methods** (SMS, webhooks, etc.)
4. **Implement multi-camera support**

## 📄 License

This project is part of your fire detection system. Use responsibly and ensure proper safety measures are in place.

## ⚠️ Safety Notice

This system is for detection purposes only. Always have proper fire safety measures and emergency procedures in place. The system should not replace professional fire monitoring systems in critical environments.

---

**Fire Detection System v2.0** - Powered by YOLOv8
