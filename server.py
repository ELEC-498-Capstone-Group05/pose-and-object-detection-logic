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
r"""Example using PyCoral to run YOLO and MoveNet on two Edge TPUs in parallel.

To run this code, you must attach two Edge TPUs to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`.

Example usage:
python3 pose_and_object.py \
  --yolo_model models/object/yolo11n_edgetpu.tflite \
  --movenet_model models/pose/movenet_single_pose_lightning_ptq_edgetpu.tflite \
  --labels models/object/coco_labels.txt
"""

import argparse
import ctypes
import cv2
import numpy as np
import threading
import time
import logging
from flask import Flask, Response, jsonify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
from pose_estimator import estimate_pose, get_pose_color, TemporalActionRecognizer, get_action_color
from object_detector import ObjectDetector
import cropping_algorithm

app = Flask(__name__)

# Suppress Werkzeug request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global variables for YOLO
yolo_interpreter = None
yolo_detector = None
yolo_input_size = None
yolo_labels = {}

# Global variables for MoveNet
movenet_interpreter = None
movenet_input_size = None
movenet_input_dtype = None
movenet_input_scale = 1.0
movenet_input_zero_point = 0
_NUM_KEYPOINTS = 17
crop_region = None # Init crop region state

# Shared resources
cap = None
frame_lock = threading.Lock()
frame_cond = threading.Condition(frame_lock)
frame_seq = 0
latest_frame = None
yolo_results = None
movenet_results = None
movenet_pose_label = "Unknown"
action_recognizer = None
movenet_action_label = "Unknown"
inference_times = {'yolo': 0.0, 'movenet': 0.0}
stage_times = {
    'yolo_pre': 0.0,
    'yolo_invoke': 0.0,
    'yolo_post': 0.0,
    'movenet_pre': 0.0,
    'movenet_invoke': 0.0,
    'movenet_post': 0.0,
    'frame_encode': 0.0,
    'frame_total': 0.0,
}

def yolo_inference_thread():
    """Runs YOLO inference continuously on TPU 0."""
    global latest_frame, yolo_results, inference_times, stage_times, frame_seq, yolo_detector

    last_seq = -1
    
    while True:
        with frame_cond:
            while latest_frame is None or frame_seq == last_seq:
                frame_cond.wait(timeout=0.1)
            frame = latest_frame.copy()
            last_seq = frame_seq

        boxes, class_ids, scores, timings, meta = yolo_detector.infer(frame)
        yolo_results = (boxes, class_ids, scores, meta)
        stage_times['yolo_pre'] = timings['pre_ms']
        stage_times['yolo_invoke'] = timings['invoke_ms']
        stage_times['yolo_post'] = timings['post_ms']
        inference_times['yolo'] = timings['invoke_ms']

def movenet_inference_thread():
    """Runs MoveNet inference continuously on TPU 1."""
    global latest_frame, movenet_results, inference_times, stage_times, frame_seq, crop_region
    global movenet_pose_label, movenet_action_label

    last_seq = -1
    
    while True:
        with frame_cond:
            while latest_frame is None or frame_seq == last_seq:
                frame_cond.wait(timeout=0.1)
            frame = latest_frame.copy()
            last_seq = frame_seq

        frame_h, frame_w, _ = frame.shape

        # Determine crop region based on previous results
        if movenet_results is None:
            crop_region = cropping_algorithm.init_crop_region(frame_h, frame_w)
        else:
            # Reshape keypoints to match cropping_algorithm expectations (1, 1, 17, 3)
            prev_keypoints = movenet_results.reshape(1, 1, 17, 3)
            crop_region = cropping_algorithm.determine_crop_region(
                prev_keypoints, frame_h, frame_w
            )

        t0 = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crop and resize using the calculated region
        resized = cropping_algorithm.crop_and_resize(
            rgb_frame, crop_region, movenet_input_size
        )

        # Feed the interpreter in its expected input dtype.
        if movenet_input_dtype is None:
            input_data = resized
        elif movenet_input_dtype == np.float32:
            input_data = resized.astype(np.float32)
        elif movenet_input_dtype == np.uint8:
            input_data = resized.astype(np.uint8)
        elif movenet_input_dtype == np.int8:
            tmp = resized.astype(np.float32)
            tmp = (tmp / movenet_input_scale) + movenet_input_zero_point
            input_data = np.clip(np.rint(tmp), -128, 127).astype(np.int8)
        else:
            input_data = resized.astype(movenet_input_dtype)

        stage_times['movenet_pre'] = (time.perf_counter() - t0) * 1000

        common.set_input(movenet_interpreter, input_data)

        t1 = time.perf_counter()
        movenet_interpreter.invoke()
        invoke_ms = (time.perf_counter() - t1) * 1000
        inference_times['movenet'] = invoke_ms
        stage_times['movenet_invoke'] = invoke_ms

        t2 = time.perf_counter()
        pose_local = common.output_tensor(movenet_interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
        
        # Estimate pose from local (cropped) keypoints for better scale invariance
        movenet_pose_label = estimate_pose(pose_local)
        
        # Convert local keypoints to global coordinates
        pose_global = np.copy(pose_local)
        for idx in range(_NUM_KEYPOINTS):
            pose_global[idx, 0] = (
                crop_region['y_min'] * frame_h +
                crop_region['height'] * frame_h *
                pose_local[idx, 0]) / frame_h
            pose_global[idx, 1] = (
                crop_region['x_min'] * frame_w +
                crop_region['width'] * frame_w *
                pose_local[idx, 1]) / frame_w
        
        # Update global results for drawing
        movenet_results = pose_global
        
        # Detect action using temporal features (using global keypoints for consistency)
        if action_recognizer is not None:
             movenet_action_label = action_recognizer.update(pose_global, static_pose=movenet_pose_label)
        
        stage_times['movenet_post'] = (time.perf_counter() - t2) * 1000

def gen_frames():
    """Generator for streaming video frames with both YOLO and MoveNet results."""
    global latest_frame, yolo_results, movenet_results, frame_seq, stage_times
    
    while True:
        frame_loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        with frame_cond:
            latest_frame = frame.copy()
            frame_seq += 1
            frame_cond.notify_all()
        
        # Draw YOLO detections
        if yolo_results is not None:
            if len(yolo_results) == 4:
                boxes, class_ids, scores, meta = yolo_results
            else:
                boxes, class_ids, scores = yolo_results
                meta = None
            height, width, _ = frame.shape
            x_scale = width / yolo_input_size[0]
            y_scale = height / yolo_input_size[1]
            
            if len(boxes) > 0:
                if np.max(boxes) < 2.0:
                    boxes = boxes * np.array([yolo_input_size[0], yolo_input_size[1], 
                                             yolo_input_size[0], yolo_input_size[1]])

            for i in range(len(boxes)):
                box = boxes[i]
                class_id = class_ids[i]
                score = scores[i]
                
                cx, cy, w, h = box
                if meta and meta.get('letterbox'):
                    scale = meta['scale']
                    pad_left = meta['pad_left']
                    pad_top = meta['pad_top']
                    cx = (cx - pad_left) / scale
                    cy = (cy - pad_top) / scale
                    w = w / scale
                    h = h / scale
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)
                else:
                    x1 = int((cx - w/2) * x_scale)
                    y1 = int((cy - h/2) * y_scale)
                    x2 = int((cx + w/2) * x_scale)
                    y2 = int((cy + h/2) * y_scale)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label_name = yolo_labels.get(class_id, f"Class {class_id}")
                label = f"{label_name}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw MoveNet keypoints
        if movenet_results is not None:
            pose = movenet_results
            height, width, _ = frame.shape
            
            for i in range(_NUM_KEYPOINTS):
                y = int(pose[i][0] * height)
                x = int(pose[i][1] * width)
                confidence = pose[i][2]
                
                if confidence > 0.3:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        t_enc = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', frame)
        stage_times['frame_encode'] = (time.perf_counter() - t_enc) * 1000
        stage_times['frame_total'] = (time.perf_counter() - frame_loop_start) * 1000
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Dual TPU Inference</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f0f0f0;
            }
            .container {
                display: flex;
                gap: 20px;
                max-width: 1200px;
            }
            .video-section {
                flex: 0 0 auto;
            }
            .stats-section {
                flex: 1;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            .stat-group {
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .stat-group h3 {
                margin-top: 0;
                color: #555;
            }
            .stat-item {
                margin: 8px 0;
                font-size: 16px;
            }
            .stat-label {
                font-weight: bold;
                display: inline-block;
                width: 180px;
            }
            .stat-value {
                color: #007bff;
            }
            .pose-value {
                font-size: 20px;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 4px;
                display: inline-block;
            }
            .action-value {
                font-size: 20px;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 4px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <h1>Dual Edge TPU Inference</h1>
        <p>YOLO Object Detection + MoveNet Pose Estimation + Action Recognition</p>
        <div class="container">
            <div class="video-section">
                <img src='/video_feed' width='640' height='480' />
            </div>
            <div class="stats-section">
                <div class="stat-group">
                    <h3>Pose & Action Detection</h3>
                    <div class="stat-item">
                        <span class="stat-label">Current Pose:</span>
                        <span class="pose-value" id="pose">Unknown</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Current Action:</span>
                        <span class="action-value" id="action">Unknown</span>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>Inference Times</h3>
                    <div class="stat-item">
                        <span class="stat-label">YOLO Inference:</span>
                        <span class="stat-value" id="yolo-time">0.00 ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">MoveNet Inference:</span>
                        <span class="stat-value" id="movenet-time">0.00 ms</span>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>Detailed Timing (YOLO)</h3>
                    <div class="stat-item">
                        <span class="stat-label">Preprocessing:</span>
                        <span class="stat-value" id="yolo-pre">0.0 ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">TPU Invocation:</span>
                        <span class="stat-value" id="yolo-invoke">0.0 ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Postprocessing:</span>
                        <span class="stat-value" id="yolo-post">0.0 ms</span>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>Detailed Timing (MoveNet)</h3>
                    <div class="stat-item">
                        <span class="stat-label">Preprocessing:</span>
                        <span class="stat-value" id="movenet-pre">0.0 ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">TPU Invocation:</span>
                        <span class="stat-value" id="movenet-invoke">0.0 ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Postprocessing:</span>
                        <span class="stat-value" id="movenet-post">0.0 ms</span>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>Frame Processing</h3>
                    <div class="stat-item">
                        <span class="stat-label">Frame Rate:</span>
                        <span class="stat-value" id="frame-rate">0.0 FPS</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Frame Encoding:</span>
                        <span class="stat-value" id="frame-encode">0.0 ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Frame Time:</span>
                        <span class="stat-value" id="frame-total">0.0 ms</span>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function updateStats() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        // Update pose and action
                        const poseElem = document.getElementById('pose');
                        poseElem.textContent = data.pose;
                        poseElem.style.backgroundColor = rgbToHex(data.pose_color);
                        
                        const actionElem = document.getElementById('action');
                        actionElem.textContent = data.action;
                        actionElem.style.backgroundColor = rgbToHex(data.action_color);
                        
                        // Update inference times
                        document.getElementById('yolo-time').textContent = data.yolo_time.toFixed(2) + ' ms';
                        document.getElementById('movenet-time').textContent = data.movenet_time.toFixed(2) + ' ms';
                        
                        // Update detailed timing
                        document.getElementById('yolo-pre').textContent = data.yolo_pre.toFixed(1) + ' ms';
                        document.getElementById('yolo-invoke').textContent = data.yolo_invoke.toFixed(1) + ' ms';
                        document.getElementById('yolo-post').textContent = data.yolo_post.toFixed(1) + ' ms';
                        
                        document.getElementById('movenet-pre').textContent = data.movenet_pre.toFixed(1) + ' ms';
                        document.getElementById('movenet-invoke').textContent = data.movenet_invoke.toFixed(1) + ' ms';
                        document.getElementById('movenet-post').textContent = data.movenet_post.toFixed(1) + ' ms';
                        
                        document.getElementById('frame-encode').textContent = data.frame_encode.toFixed(1) + ' ms';
                        document.getElementById('frame-total').textContent = data.frame_total.toFixed(1) + ' ms';
                        document.getElementById('frame-rate').textContent = data.fps.toFixed(1) + ' FPS';
                    })
                    .catch(error => console.error('Error fetching stats:', error));
            }
            
            function rgbToHex(rgb) {
                // Convert BGR tuple to hex color
                const [b, g, r] = rgb;
                return '#' + [r, g, b].map(x => {
                    const hex = x.toString(16);
                    return hex.length === 1 ? '0' + hex : hex;
                }).join('');
            }
            
            // Update stats every 100ms
            setInterval(updateStats, 100);
            updateStats();
        </script>
    </body>
    </html>
    """

@app.route('/stats')
def stats():
    """Return current inference statistics as JSON."""
    from flask import jsonify
    fps = 1000 / stage_times['frame_total'] if stage_times['frame_total'] > 0 else 0
    return jsonify({
        'pose': movenet_pose_label,
        'pose_color': get_pose_color(movenet_pose_label),
        'action': movenet_action_label,
        'action_color': get_action_color(movenet_action_label),
        'yolo_time': inference_times['yolo'],
        'movenet_time': inference_times['movenet'],
        'yolo_pre': stage_times['yolo_pre'],
        'yolo_invoke': stage_times['yolo_invoke'],
        'yolo_post': stage_times['yolo_post'],
        'movenet_pre': stage_times['movenet_pre'],
        'movenet_invoke': stage_times['movenet_invoke'],
        'movenet_post': stage_times['movenet_post'],
        'frame_encode': stage_times['frame_encode'],
        'frame_total': stage_times['frame_total'],
        'fps': fps
    })

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    global yolo_interpreter, yolo_detector, yolo_input_size, yolo_labels
    global movenet_interpreter, movenet_input_size, movenet_input_dtype
    global movenet_input_scale, movenet_input_zero_point, cap
    global action_recognizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', required=True, help='Path to YOLO .tflite model')
    parser.add_argument('--movenet_model', required=True, help='Path to MoveNet .tflite model')
    parser.add_argument('--labels', help='Path to COCO labels file')
    parser.add_argument('--camera_idx', type=int, default=0, help='Camera index')
    parser.add_argument('--yolo_debug', action='store_true', help='Enable YOLO input/output debug dumps')
    parser.add_argument('--yolo_debug_dir', default='debug_yolo', help='Directory for YOLO debug images')
    parser.add_argument('--yolo_debug_every', type=int, default=60, help='Dump YOLO debug info every N frames')
    args = parser.parse_args()

    if args.yolo_debug:
        logging.basicConfig(level=logging.INFO)

    # Fail fast with a clearer error if the Edge TPU runtime is missing.
    try:
        ctypes.CDLL('libedgetpu.so.1')
    except OSError as e:
        print('Error: failed to load Edge TPU runtime library libedgetpu.so.1')
        print(f'Details: {e}')
        print('Fix: install the Edge TPU runtime (libedgetpu) for your OS/arch, then retry.')
        print('Quick checks:')
        print("  - ldconfig -p | grep edgetpu")
        print("  - ls -l /usr/lib/*/libedgetpu.so.1")
        return

    # Load labels
    if args.labels:
        yolo_labels = read_label_file(args.labels)

    # Discover available Edge TPUs.
    tpus = list_edge_tpus()
    if not tpus:
        print('Error: no Edge TPUs found. Check USB connection and power.')
        return

    # Prefer the canonical USB device spec (usb:N). Some environments do not accept ':N'.
    candidate_devices = [f"usb:{i}" for i in range(len(tpus))]

    def _make_on_first_available(model_path: str, exclude: set[str]):
        last_error: Exception | None = None
        for dev in candidate_devices:
            if dev in exclude:
                continue
            try:
                return dev, make_interpreter(model_path, device=dev)
            except Exception as e:
                last_error = e
        raise RuntimeError(f"No usable Edge TPU device found for model {model_path}. Last error: {last_error}")

    # Initialize YOLO interpreter on TPU 0
    try:
        yolo_device, yolo_interpreter = _make_on_first_available(args.yolo_model, exclude=set())
    except Exception as e:
        print('Error: failed to create YOLO Edge TPU interpreter (delegate load failed).')
        print(f'Details: {e}')
        print('This usually means libedgetpu is missing/mismatched, device permissions are wrong, or a TPU is not usable.')
        print(f"Detected TPUs: {tpus}")
        return
    print(f"Initializing YOLO interpreter on {yolo_device}...")
    yolo_interpreter.allocate_tensors()
    yolo_detector = ObjectDetector(
        yolo_interpreter,
        labels=yolo_labels,
        debug=args.yolo_debug,
        debug_dir=args.yolo_debug_dir,
        debug_every=args.yolo_debug_every,
    )
    yolo_input_size = yolo_detector.input_size
    yolo_labels = yolo_detector.labels
    
    print(f"YOLO model loaded. Input size: {yolo_input_size}")

    # Initialize MoveNet interpreter on TPU 1
    try:
        movenet_device, movenet_interpreter = _make_on_first_available(
            args.movenet_model, exclude={yolo_device}
        )
    except Exception as e:
        print('Error: failed to create MoveNet Edge TPU interpreter (delegate load failed).')
        print(f'Details: {e}')
        print('This usually means libedgetpu is missing/mismatched, device permissions are wrong, or a TPU is not usable.')
        print(f"Detected TPUs: {tpus}")
        print(f"YOLO is using: {yolo_device}")
        return
    print(f"Initializing MoveNet interpreter on {movenet_device}...")
    movenet_interpreter.allocate_tensors()
    movenet_input_size = common.input_size(movenet_interpreter)

    movenet_input_details = movenet_interpreter.get_input_details()[0]
    movenet_input_dtype = movenet_input_details.get('dtype', None)
    q = movenet_input_details.get('quantization', (1.0, 0))
    movenet_input_scale = q[0] if q and len(q) > 0 else 1.0
    movenet_input_zero_point = q[1] if q and len(q) > 1 else 0
    
    print(f"MoveNet model loaded. Input size: {movenet_input_size}")
    
    # Initialize crop region
    global crop_region
    # We don't have image dims yet, so leave as None, will init in loop
    crop_region = None

    # Open camera - try multiple indices if the specified one fails
    camera_indices_to_try = [args.camera_idx] + [i for i in range(10) if i != args.camera_idx]
    cap = None
    successful_camera_idx = None
    
    for idx in camera_indices_to_try:
        print(f"Trying camera index {idx}...")
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            # Verify we can actually read a frame
            ret, _ = test_cap.read()
            if ret:
                cap = test_cap
                successful_camera_idx = idx
                print(f"Successfully opened camera {idx}")
                break
            else:
                test_cap.release()
        else:
            test_cap.release()
    
    if cap is None or not cap.isOpened():
        print(f"Error: Could not open any camera (tried indices: {camera_indices_to_try[:5]})")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if not available
    print(f"Camera {successful_camera_idx} opened: {width}x{height} @ {fps} FPS")
    
    # Initialize temporal action recognizer
    action_recognizer = TemporalActionRecognizer(window_size=30, fps=fps)
    print(f"Temporal action recognizer initialized (buffer size: 30 frames)")
    
    # Start inference threads
    print("Starting inference threads...")
    yolo_thread = threading.Thread(target=yolo_inference_thread, daemon=True)
    movenet_thread = threading.Thread(target=movenet_inference_thread, daemon=True)
    
    yolo_thread.start()
    movenet_thread.start()
    
    print("Starting Flask server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()