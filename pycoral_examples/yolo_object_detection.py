# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to detect objects with YOLO model and stream via Flask.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`.

Example usage:
python3 yolo_object_detection.py \
  --model models/object/yolo11n_edgetpu.tflite \
    --labels models/object/coco_labels.txt
"""

import argparse
import cv2
import numpy as np
from PIL import Image
from flask import Flask, Response
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

app = Flask(__name__)

interpreter = None
cap = None
input_size = None
labels = {}
input_scale = 1.0

def non_max_suppression(boxes, scores, threshold):
    """Performs Non-Maximum Suppression (NMS) on the boxes."""
    if len(boxes) == 0:
        return []

    # Convert to x1, y1, x2, y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

def process_yolo_output(output, conf_threshold=0.25, iou_threshold=0.45):
    """Processes YOLO output tensor."""
    # Output shape is typically [1, 4 + num_classes, num_anchors]
    # Transpose to [1, num_anchors, 4 + num_classes]
    output = output.transpose((0, 2, 1))
    
    # Remove batch dimension
    pred = output[0]
    
    # Split boxes and scores
    boxes = pred[:, :4]
    scores = pred[:, 4:]
    
    # Get max score and class index for each anchor
    class_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)
    
    # Filter by confidence
    mask = max_scores > conf_threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]
    
    if len(boxes) == 0:
        return [], [], []

    # Apply NMS
    keep = non_max_suppression(boxes, max_scores, iou_threshold)
    
    return boxes[keep], class_ids[keep], max_scores[keep]

import time

# ...existing code...

def gen_frames():
    global cap, interpreter, input_size, input_scale, input_zero_point, output_scale, output_zero_point
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        resized_img = pil_img.resize(input_size, Image.NEAREST)
        
        # Quantize input
        # Model expects normalized 0-1 float converted to int8
        # q = (real / scale) + zero_point
        # real = pixel / 255.0
        input_data = np.array(resized_img, dtype=np.float32) / 255.0
        input_data = (input_data / input_scale) + input_zero_point
        input_data = input_data.astype(np.int8)
        
        # Run inference
        common.set_input(interpreter, input_data)
        
        start_time = time.perf_counter()
        interpreter.invoke()
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Get output and dequantize
        output = common.output_tensor(interpreter, 0).copy()
        output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        # Process YOLO output
        boxes, class_ids, scores = process_yolo_output(output)
        
        # Draw detections
        height, width, _ = frame.shape
        x_scale = width / input_size[0]
        y_scale = height / input_size[1]
        
        if len(boxes) > 0:
            # Check if boxes are normalized (0-1) or pixels (0-640)
            # Based on quantization scale ~0.004, they are likely normalized
            if np.max(boxes) < 2.0:
                boxes = boxes * np.array([input_size[0], input_size[1], input_size[0], input_size[1]])

        for i in range(len(boxes)):
            box = boxes[i]
            class_id = class_ids[i]
            score = scores[i]
            
            # Convert cx, cy, w, h to x1, y1, x2, y2 and scale
            cx, cy, w, h = box
            x1 = int((cx - w/2) * x_scale)
            y1 = int((cy - h/2) * y_scale)
            x2 = int((cx + w/2) * x_scale)
            y2 = int((cy + h/2) * y_scale)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_name = labels.get(class_id, f"Class {class_id}")
            label = f"{label_name}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw inference time
        cv2.putText(frame, f"Inference: {inference_time:.2f}ms", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return "<body><h1>YOLO Object Detection Live Feed</h1><img src='/video_feed' width='640' height='480' /></body>"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    global interpreter, cap, input_size, input_scale, input_zero_point, output_scale, output_zero_point, labels
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .tflite model')
    parser.add_argument('--labels', help='Path to label file')
    parser.add_argument('--camera_idx', type=int, default=0, help='Camera index')
    args = parser.parse_args()

    # Load labels
    if args.labels:
        labels = read_label_file(args.labels)

    # Initialize interpreter
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    input_size = common.input_size(interpreter)
    
    # Get quantization details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale = input_details['quantization'][0]
    input_zero_point = input_details['quantization'][1]
    output_scale = output_details['quantization'][0]
    output_zero_point = output_details['quantization'][1]
    
    print(f"Model loaded. Input size: {input_size}")
    print(f"Input Quantization: scale={input_scale}, zero_point={input_zero_point}")
    print(f"Output Quantization: scale={output_scale}, zero_point={output_zero_point}")

    # Initialize camera
    cap = cv2.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_idx}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {width}x{height}")
    
    print("Starting Flask server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
