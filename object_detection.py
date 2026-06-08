import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained SSD model from TensorFlow Hub
import tensorflow_hub as hub
detector = hub.load(
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
)

# Load label map for COCO dataset
labels_path = tf.keras.utils.get_file('mscoco_label_map.txt',
                                      'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt')
labels = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light"}  # simplified

def preprocess_frame(frame):
    img = cv2.resize(frame, (320, 320))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
    return tf.expand_dims(img_tensor, 0)

def draw_boxes(frame, boxes, class_ids, scores, threshold=0.5):
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, top, right, bottom) = (xmin * width, ymin * height, xmax * width, ymax * height)
            class_id = int(class_ids[i])
            label = labels.get(class_id, 'Unknown')
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {scores[i]:.2f}", (int(left), int(top)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

# Real-time webcam feed
cap = cv2.VideoCapture(0)

print("Starting real-time object detection. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess_frame(frame)
    outputs = detector(input_tensor)

    # Convert tensors to numpy arrays
    boxes = outputs['detection_boxes'][0].numpy()
    class_ids = outputs['detection_classes'][0].numpy()
    scores = outputs['detection_scores'][0].numpy()

    # Draw detection results
    frame = draw_boxes(frame, boxes, class_ids, scores)

    cv2.imshow("SSD Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
