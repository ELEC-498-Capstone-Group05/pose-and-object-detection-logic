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
python3 server.py \
  --yolo_model models/object/yolo26n_full_integer_quant_edgetpu.tflite \
  --movenet_model models/pose/movenet_single_pose_thunder_ptq_edgetpu.tflite \
  --labels models/object/coco_labels.txt
"""

import argparse
import ctypes
import cv2
import numpy as np
import threading
import time
import logging
from collections import deque
from flask import Flask, Response, jsonify, render_template, send_file, abort, request
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
from pose_estimator import estimate_pose, get_pose_color, TemporalActionRecognizer, get_action_color
from object_detector import ObjectDetector, ObjectTracker
import cropping_algorithm
from alert_system import AlertSystem
from audio_classifier import AudioClassifier
from video_recorder import RollingVideoRecorder

app = Flask(__name__)
logger = logging.getLogger(__name__)
alert_system = AlertSystem(app)

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
yolo_person_present = False
yolo_result_seq = -1
movenet_results = None
movenet_pose_label = "Unknown"
action_recognizer = None
movenet_action_label = "Unknown"
movenet_result_seq = -1
audio_classifier = None
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
video_recorder = None

process_start_unix_ms = int(time.time() * 1000)
frame_capture_ts = {}
MAX_CAPTURE_TS = 600

metrics_lock = threading.Lock()
metrics = {
    'capture_fps': 0.0,
    'yolo_fps': 0.0,
    'movenet_fps': 0.0,
    'stream_fps': 0.0,
    'yolo_lag_frames': 0,
    'movenet_lag_frames': 0,
    'yolo_lag_p95': 0.0,
    'movenet_lag_p95': 0.0,
    'capture_to_stream_latency_ms': 0.0,
    'capture_to_stream_latency_p95_ms': 0.0,
    'yolo_dropped_frames_total': 0,
    'movenet_dropped_frames_total': 0,
    'frame_drops_total': 0,
    'yolo_result_age_frames': 0,
    'movenet_result_age_frames': 0,
    'yolo_result_age_p95_frames': 0.0,
    'movenet_result_age_p95_frames': 0.0,
}

_yolo_lag_samples = deque(maxlen=300)
_movenet_lag_samples = deque(maxlen=300)
_capture_to_stream_latency_samples = deque(maxlen=300)
_yolo_result_age_samples = deque(maxlen=300)
_movenet_result_age_samples = deque(maxlen=300)


def _is_person_class_id(class_id):
    """Returns True if class_id corresponds to a person label."""
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False

    label = yolo_labels.get(cid)
    if isinstance(label, str) and label.strip().lower() == 'person':
        return True

    # COCO fallback for person class.
    return cid == 0


def _has_person_detection(class_ids):
    """Returns True when YOLO detections include at least one person."""
    if class_ids is None or len(class_ids) == 0:
        return False

    for class_id in class_ids:
        if _is_person_class_id(class_id):
            return True
    return False

def yolo_inference_thread():
    """Runs YOLO inference continuously on TPU 0."""
    global latest_frame, yolo_results, yolo_person_present, yolo_result_seq
    global inference_times, stage_times, frame_seq, yolo_detector

    tracker = ObjectTracker()
    last_seq = -1
    fps_count = 0
    fps_window_start = time.perf_counter()
    
    while True:
        with frame_cond:
            while latest_frame is None or frame_seq == last_seq:
                frame_cond.wait(timeout=0.1)
            current_seq = frame_seq
            frame = latest_frame.copy()

            # Count skipped frames for this consumer thread.
            if last_seq >= 0 and current_seq > last_seq + 1:
                dropped = current_seq - last_seq - 1
                with metrics_lock:
                    metrics['yolo_dropped_frames_total'] += dropped
                    metrics['frame_drops_total'] += dropped

            last_seq = current_seq

        boxes, class_ids, scores, timings, meta = yolo_detector.infer(frame)
        tracked_boxes, tracked_class_ids, tracked_scores = tracker.update(boxes, class_ids, scores)
        person_present = _has_person_detection(tracked_class_ids)

        with frame_lock:
            yolo_results = (tracked_boxes, tracked_class_ids, tracked_scores, meta)
            yolo_person_present = person_present
            yolo_result_seq = current_seq

        with frame_lock:
            latest_seq_snapshot = frame_seq
        yolo_lag = max(0, latest_seq_snapshot - current_seq)
        _yolo_lag_samples.append(yolo_lag)

        fps_count += 1
        if fps_count >= 30:
            now = time.perf_counter()
            elapsed = now - fps_window_start
            yolo_fps = (fps_count / elapsed) if elapsed > 0 else 0.0
            with metrics_lock:
                metrics['yolo_fps'] = yolo_fps
                metrics['yolo_lag_frames'] = yolo_lag
                if _yolo_lag_samples:
                    metrics['yolo_lag_p95'] = float(np.percentile(np.array(_yolo_lag_samples), 95))
            fps_count = 0
            fps_window_start = now
        else:
            with metrics_lock:
                metrics['yolo_lag_frames'] = yolo_lag

        stage_times['yolo_pre'] = timings['pre_ms']
        stage_times['yolo_invoke'] = timings['invoke_ms']
        stage_times['yolo_post'] = timings['post_ms']
        inference_times['yolo'] = timings['invoke_ms']

def movenet_inference_thread():
    """Runs MoveNet inference continuously on TPU 1."""
    global latest_frame, movenet_results, inference_times, stage_times, frame_seq, crop_region
    global movenet_pose_label, movenet_action_label
    global yolo_person_present, movenet_result_seq

    last_seq = -1
    fps_count = 0
    fps_window_start = time.perf_counter()
    
    while True:
        with frame_cond:
            while latest_frame is None or frame_seq == last_seq:
                frame_cond.wait(timeout=0.1)
            current_seq = frame_seq
            frame = latest_frame.copy()
            capture_ts = frame_capture_ts.get(current_seq)

            if last_seq >= 0 and current_seq > last_seq + 1:
                dropped = current_seq - last_seq - 1
                with metrics_lock:
                    metrics['movenet_dropped_frames_total'] += dropped
                    metrics['frame_drops_total'] += dropped

            last_seq = current_seq
            person_present = yolo_person_present

        if not person_present:
            with frame_lock:
                movenet_results = None
                movenet_pose_label = "Unknown"
                movenet_action_label = "Unknown"
                crop_region = None
                movenet_result_seq = -1
                inference_times['movenet'] = 0.0
                stage_times['movenet_pre'] = 0.0
                stage_times['movenet_invoke'] = 0.0
                stage_times['movenet_post'] = 0.0
            continue

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
        movenet_result_seq = current_seq
        
        # Detect action using temporal features (using global keypoints for consistency)
        if action_recognizer is not None:
             movenet_action_label = action_recognizer.update(
                 pose_global,
                 static_pose=movenet_pose_label,
                 frame_seq=current_seq,
                 capture_ts=capture_ts,
             )
        
        stage_times['movenet_post'] = (time.perf_counter() - t2) * 1000

        with frame_lock:
            latest_seq_snapshot = frame_seq
        movenet_lag = max(0, latest_seq_snapshot - current_seq)
        _movenet_lag_samples.append(movenet_lag)

        fps_count += 1
        if fps_count >= 30:
            now = time.perf_counter()
            elapsed = now - fps_window_start
            movenet_fps = (fps_count / elapsed) if elapsed > 0 else 0.0
            with metrics_lock:
                metrics['movenet_fps'] = movenet_fps
                metrics['movenet_lag_frames'] = movenet_lag
                if _movenet_lag_samples:
                    metrics['movenet_lag_p95'] = float(np.percentile(np.array(_movenet_lag_samples), 95))
            fps_count = 0
            fps_window_start = now
        else:
            with metrics_lock:
                metrics['movenet_lag_frames'] = movenet_lag

def annotate_frame(frame, yolo_results, movenet_results, yolo_input_size, yolo_labels):
    """Draws YOLO and MoveNet results on the frame."""
    global _NUM_KEYPOINTS
    
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

def camera_loop():
    """Continuously captures frames from the camera and notifies inference threads."""
    global latest_frame, frame_seq, yolo_results, movenet_results 
    global movenet_action_label, yolo_input_size, cap, video_recorder

    fps_count = 0
    fps_window_start = time.perf_counter()
    
    while True:
        if cap is None:
            time.sleep(1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        if video_recorder is not None:
            video_recorder.write(frame)

        with frame_cond:
            latest_frame = frame.copy()
            frame_seq += 1
            frame_capture_ts[frame_seq] = time.perf_counter()
            while len(frame_capture_ts) > MAX_CAPTURE_TS:
                frame_capture_ts.pop(next(iter(frame_capture_ts)))
            frame_cond.notify_all()

        fps_count += 1
        if fps_count >= 30:
            now = time.perf_counter()
            elapsed = now - fps_window_start
            with metrics_lock:
                metrics['capture_fps'] = (fps_count / elapsed) if elapsed > 0 else 0.0
            fps_count = 0
            fps_window_start = now
            
        # Run alert processing on every frame if we have yolo input size set up
        # This ensures alerts work even if no video client is connected
        if yolo_input_size is not None:
            alert_system.process_frame(
                yolo_results, 
                movenet_results, 
                movenet_action_label, 
                yolo_input_size
            )

def gen_frames():
    """Generator for streaming video frames with both YOLO and MoveNet results."""
    global latest_frame, yolo_results, movenet_results, frame_seq, stage_times
    global yolo_result_seq, movenet_result_seq
    global yolo_input_size, yolo_labels
    
    last_seq = -1
    fps_count = 0
    fps_window_start = time.perf_counter()
    
    while True:
        with frame_cond:
            while latest_frame is None or frame_seq == last_seq:
                frame_cond.wait(timeout=0.1)
            frame = latest_frame.copy()
            current_seq = frame_seq
            last_seq = current_seq

        with frame_lock:
            yolo_seq_snapshot = yolo_result_seq
            movenet_seq_snapshot = movenet_result_seq
            
        t_start = time.perf_counter()
        
        # Draw annotations
        annotate_frame(frame, yolo_results, movenet_results, yolo_input_size, yolo_labels)

        t_enc = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', frame)
        stage_times['frame_encode'] = (time.perf_counter() - t_enc) * 1000
        stage_times['frame_total'] = (time.perf_counter() - t_start) * 1000

        with frame_cond:
            capture_ts = frame_capture_ts.pop(current_seq, None)
        if capture_ts is not None:
            capture_to_stream_ms = (time.perf_counter() - capture_ts) * 1000.0
            _capture_to_stream_latency_samples.append(capture_to_stream_ms)
            with metrics_lock:
                metrics['capture_to_stream_latency_ms'] = capture_to_stream_ms
                if _capture_to_stream_latency_samples:
                    metrics['capture_to_stream_latency_p95_ms'] = float(
                        np.percentile(np.array(_capture_to_stream_latency_samples), 95)
                    )

        yolo_result_age = max(0, current_seq - yolo_seq_snapshot) if yolo_seq_snapshot >= 0 else 0
        movenet_result_age = max(0, current_seq - movenet_seq_snapshot) if movenet_seq_snapshot >= 0 else 0
        _yolo_result_age_samples.append(yolo_result_age)
        _movenet_result_age_samples.append(movenet_result_age)
        with metrics_lock:
            metrics['yolo_result_age_frames'] = yolo_result_age
            metrics['movenet_result_age_frames'] = movenet_result_age
            if _yolo_result_age_samples:
                metrics['yolo_result_age_p95_frames'] = float(np.percentile(np.array(_yolo_result_age_samples), 95))
            if _movenet_result_age_samples:
                metrics['movenet_result_age_p95_frames'] = float(np.percentile(np.array(_movenet_result_age_samples), 95))

        fps_count += 1
        if fps_count >= 30:
            now = time.perf_counter()
            elapsed = now - fps_window_start
            with metrics_lock:
                metrics['stream_fps'] = (fps_count / elapsed) if elapsed > 0 else 0.0
            fps_count = 0
            fps_window_start = now
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats')
def stats():
    """Return current inference statistics as JSON."""
    from flask import jsonify
    now_unix_ms = int(time.time() * 1000)
    with metrics_lock:
        metrics_snapshot = dict(metrics)
    processing_fps = 1000 / stage_times['frame_total'] if stage_times['frame_total'] > 0 else 0.0
    fps = metrics_snapshot['stream_fps'] if metrics_snapshot['stream_fps'] > 0 else processing_fps
    recorder_info = video_recorder.get_storage_info() if video_recorder is not None else {'enabled': False}
    return jsonify({
        'schema_version': '2.1.0',
        'timestamp_unix_ms': now_unix_ms,
        'process_start_unix_ms': process_start_unix_ms,
        'pose': movenet_pose_label,
        'pose_color': get_pose_color(movenet_pose_label),
        'action': movenet_action_label,
        'action_color': get_action_color(movenet_action_label),
        'audio_label': audio_classifier.latest_top_label if audio_classifier and hasattr(audio_classifier, 'latest_top_label') else "Off",
        'audio_db': audio_classifier.latest_db if audio_classifier and hasattr(audio_classifier, 'latest_db') else -100.0,
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
        'fps': fps,
        'processing_fps': processing_fps,
        'latency': {
            'frame_total_ms': stage_times['frame_total'],
            'frame_encode_ms': stage_times['frame_encode'],
            'capture_to_stream_ms': metrics_snapshot['capture_to_stream_latency_ms'],
            'capture_to_stream_p95_ms': metrics_snapshot['capture_to_stream_latency_p95_ms'],
            # Deprecated aliases maintained for backward compatibility.
            'end_to_end_ms': metrics_snapshot['capture_to_stream_latency_ms'],
            'end_to_end_p95_ms': metrics_snapshot['capture_to_stream_latency_p95_ms'],
            'yolo_invoke_ms': inference_times['yolo'],
            'movenet_invoke_ms': inference_times['movenet'],
        },
        'throughput': {
            'capture_fps': metrics_snapshot['capture_fps'],
            'yolo_fps': metrics_snapshot['yolo_fps'],
            'movenet_fps': metrics_snapshot['movenet_fps'],
            'stream_output_fps': metrics_snapshot['stream_fps'] if metrics_snapshot['stream_fps'] > 0 else fps,
            'frame_drops_total': metrics_snapshot['frame_drops_total'],
            'yolo_dropped_frames_total': metrics_snapshot['yolo_dropped_frames_total'],
            'movenet_dropped_frames_total': metrics_snapshot['movenet_dropped_frames_total'],
        },
        'queue': {
            'yolo_lag_frames': metrics_snapshot['yolo_lag_frames'],
            'yolo_lag_p95_frames': metrics_snapshot['yolo_lag_p95'],
            'movenet_lag_frames': metrics_snapshot['movenet_lag_frames'],
            'movenet_lag_p95_frames': metrics_snapshot['movenet_lag_p95'],
            'yolo_result_age_frames': metrics_snapshot['yolo_result_age_frames'],
            'yolo_result_age_p95_frames': metrics_snapshot['yolo_result_age_p95_frames'],
            'movenet_result_age_frames': metrics_snapshot['movenet_result_age_frames'],
            'movenet_result_age_p95_frames': metrics_snapshot['movenet_result_age_p95_frames'],
        },
        'recording': recorder_info
    })

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clips')
def list_clips():
    if video_recorder is None:
        return jsonify({'clips': [], 'storage': {'enabled': False}})

    return jsonify({
        'clips': video_recorder.list_clips(),
        'storage': video_recorder.get_storage_info()
    })


@app.route('/clips/<path:filename>')
def download_clip(filename):
    if video_recorder is None:
        abort(404)

    clip_path = video_recorder.get_clip_path(filename)
    if clip_path is None:
        abort(404)

    return send_file(clip_path, as_attachment=True, download_name=filename)


@app.route('/safezone', methods=['GET'])
def get_safezone():
    return jsonify({
        'ok': True,
        'safezone': alert_system.get_safezone_config(),
        'errors': None
    })


@app.route('/safezone', methods=['POST'])
def set_safezone():
    payload = request.get_json(silent=True)
    config, errors = alert_system.update_safezone_config(payload)

    if errors is not None:
        return jsonify({
            'ok': False,
            'safezone': None,
            'errors': errors
        }), 400

    return jsonify({
        'ok': True,
        'safezone': config,
        'errors': None
    })

def main():
    global yolo_interpreter, yolo_detector, yolo_input_size, yolo_labels
    global movenet_interpreter, movenet_input_size, movenet_input_dtype
    global movenet_input_scale, movenet_input_zero_point, cap
    global action_recognizer, audio_classifier, video_recorder
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', required=True, help='Path to YOLO .tflite model')
    parser.add_argument('--movenet_model', required=True, help='Path to MoveNet .tflite model')
    parser.add_argument('--labels', help='Path to COCO labels file')
    parser.add_argument('--camera_idx', type=int, default=0, help='Camera index')
    parser.add_argument('--yolo_debug', action='store_true', help='Enable YOLO input/output debug dumps')
    parser.add_argument('--yolo_debug_dir', default='debug_yolo', help='Directory for YOLO debug images')
    parser.add_argument('--yolo_debug_every', type=int, default=60, help='Dump YOLO debug info every N frames')
    parser.add_argument('--mic', type=int, default=None, help='Microphone device index')
    parser.add_argument('--recordings_dir', default='recordings', help='Directory where video clips are stored')
    parser.add_argument('--record_clip_seconds', type=int, default=60, help='Length of each recorded clip in seconds')
    parser.add_argument('--recording_max_gb', type=float, default=80.0, help='Maximum storage budget for recordings (GB)')
    args = parser.parse_args()

    # Configure logging to always capture INFO and above messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Logging configured. Starting Child Monitor...")

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

    video_recorder = RollingVideoRecorder(
        output_dir=args.recordings_dir,
        clip_seconds=args.record_clip_seconds,
        max_storage_gb=args.recording_max_gb,
    )
    video_recorder.start(width=width, height=height, fps=fps)
    print(
        f"Video recording enabled: {args.record_clip_seconds}s clips, "
        f"max {args.recording_max_gb:.1f} GB, dir='{args.recordings_dir}'"
    )
    
    # Initialize temporal action recognizer
    action_recognizer = TemporalActionRecognizer(window_size=30, fps=fps)
    print(f"Temporal action recognizer initialized (buffer size: 30 frames)")
    
    # Initialize Audio Classifier
    audio_classifier = AudioClassifier(
        model_path='models/audio/yamnet.tflite',
        labels_path='models/audio/yamnet_class_map.csv',
        callback=alert_system.process_audio,
        device_index=args.mic
    )
    
    audio_thread = threading.Thread(target=audio_classifier.run, daemon=True)
    audio_thread.start()
    print("Audio monitoring thread started")
    
    print("Starting inference threads...")
    
    # Start the camera loop thread
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    print("Camera capture thread started")

    yolo_thread = threading.Thread(target=yolo_inference_thread, daemon=True)
    movenet_thread = threading.Thread(target=movenet_inference_thread, daemon=True)
    
    yolo_thread.start()
    movenet_thread.start()
    
    print("Starting Flask server with SocketIO at http://0.0.0.0:5000")
    try:
        alert_system.socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    finally:
        if audio_classifier is not None:
            audio_classifier.stop()
        if video_recorder is not None:
            video_recorder.close()
        if cap is not None:
            cap.release()

if __name__ == '__main__':
    main()