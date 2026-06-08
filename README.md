# Real-Time Object Detection Using SSD

A Python application for real-time object detection using TensorFlow's pre-trained SSD MobileNet v2 model from TensorFlow Hub. This project detects objects in real-time from a webcam feed and displays bounding boxes with labels and confidence scores.

## Features

- **Real-time Detection**: Live object detection from webcam feed
- **Pre-trained Model**: Uses SSD MobileNet v2 model from TensorFlow Hub
- **COCO Dataset**: Supports 90 object classes from Microsoft COCO dataset
- **Confidence Filtering**: Displays only detections above a configurable threshold (default: 0.5)
- **Bounding Boxes**: Visualizes detected objects with labeled boxes and confidence scores

## Project Structure

```
Real-Time-Object-Detection-Using-SSD/
│
├── object_detection.py      # Main detection script
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── screenshots/            # Sample detection outputs
│   ├── detection1.png
│   └── detection2.png
└── .gitignore             # Git ignore rules
```

## Prerequisites

- Python 3.7 or higher
- Webcam connected to your system
- TensorFlow 2.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yash-Tummala/Object_detection.git
cd Object_detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the object detection script:
```bash
python object_detection.py
```

**Controls:**
- Press `q` to quit the application

The script will:
1. Access your webcam
2. Capture frames in real-time
3. Resize frames to 320x320 (model input size)
4. Run object detection on each frame
5. Display bounding boxes with class labels and confidence scores
6. Exit when you press 'q'

## How It Works

### Model Details
- **Model**: SSD MobileNet v2 FPNLite
- **Input Size**: 320x320 pixels
- **Dataset**: COCO (Common Objects in Context)
- **Number of Classes**: 90 object categories

### Processing Pipeline

1. **Frame Capture**: Read frame from webcam
2. **Preprocessing**: 
   - Resize to 320x320
   - Convert BGR to RGB
   - Convert to tensor format
3. **Detection**: Run through SSD MobileNet v2 model
4. **Post-processing**:
   - Extract bounding boxes, class IDs, and confidence scores
   - Filter by confidence threshold (0.5)
5. **Visualization**: Draw boxes and labels on frame
6. **Display**: Show annotated frame in window

### Key Functions

- **`preprocess_frame(frame)`**: Prepares input frames for the model
  - Resizes to 320x320
  - Converts color space
  - Creates tensor

- **`draw_boxes(frame, boxes, class_ids, scores, threshold)`**: Draws detections on frame
  - Filters by confidence threshold
  - Converts normalized coordinates to pixel values
  - Draws rectangles and labels

## Performance Considerations

- **Speed**: Real-time processing on most modern systems
- **Accuracy**: Good balance between speed and accuracy
- **Resource Usage**: Lightweight model suitable for laptops and edge devices

### Optimization Tips

- Adjust confidence threshold for fewer/more detections
- Process every N-th frame for faster performance on slower hardware
- Use GPU acceleration if available

## Supported Object Classes

The model can detect 90 different object classes from the COCO dataset, including:
- People and body parts
- Animals (dogs, cats, birds, etc.)
- Vehicles (cars, trucks, bicycles, etc.)
- Indoor objects (furniture, appliances, etc.)
- Sports equipment
- And many more...

## Configuration

You can modify the following parameters in `object_detection.py`:

```python
# Confidence threshold (0.0 to 1.0)
threshold = 0.5

# Input size (model expects 320x320)
input_size = (320, 320)

# Window name
window_name = "SSD Real-Time Detection"
```

## Troubleshooting

**Issue**: Webcam not detected
- Solution: Ensure your webcam is connected and not in use by another application

**Issue**: Slow performance
- Solution: Reduce frame resolution, skip frames, or run on GPU

**Issue**: Model download fails
- Solution: Check internet connection, TensorFlow Hub may be temporarily unavailable

**Issue**: TensorFlow errors
- Solution: Ensure correct TensorFlow version (2.x) is installed

## Dependencies

See `requirements.txt` for all required packages and versions.

## Future Enhancements

- [ ] GPU acceleration support
- [ ] Frame rate optimization
- [ ] Multiple object tracking
- [ ] Video file input support
- [ ] Configuration file support
- [ ] Performance metrics display (FPS counter)
- [ ] Adjustable confidence threshold via keyboard input

## References

- [TensorFlow Hub - SSD MobileNet v2](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1)
- [COCO Dataset](https://cocodataset.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)

## License

This project is open source and available under the MIT License.

## Author

Yash Tummala

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.
