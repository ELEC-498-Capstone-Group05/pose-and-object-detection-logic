# Child Monitor - AI Coding Assistant Instructions

## Project Overview

Real-time child safety monitoring system designed to detect hazardous or unwanted events (fighting, jumping on furniture, crying, playing with dangerous objects like knives/sockets/fire) while parents are working or preoccupied.

**Hardware**: Raspberry Pi 5 with 2× Google Coral Edge TPU accelerators + USB webcam

**Current Implementation** (parallel computer vision inference):
- **YOLO (TPU 0)**: Object detection via [object_detector.py](object_detector.py) using COCO dataset
- **MoveNet (TPU 1)**: Human pose estimation + temporal action recognition via [pose_estimator.py](pose_estimator.py)
- **Flask server**: HTTP video streaming + statistics API at [server.py](server.py)

This is an embedded/edge AI application optimized for Google Coral Edge TPU devices, not traditional cloud/GPU inference.

## Architecture & Data Flow

### Parallel Inference Pipeline
1. **Camera capture** (`gen_frames()`) reads frames, broadcasts to inference threads via `frame_cond` condition variable
2. **YOLO thread** (`yolo_inference_thread()`) continuously processes frames for object detection
3. **MoveNet thread** (`movenet_inference_thread()`) runs pose estimation with adaptive cropping
4. **Results merged** in `gen_frames()` for visualization and streaming

### Critical Dependency: Adaptive Cropping
- MoveNet uses **stateful cropping** ([cropping_algorithm.py](cropping_algorithm.py)) to track subjects across frames
- Previous frame's keypoints determine next frame's crop region for efficiency
- Global `crop_region` state initialized via `init_crop_region()`, updated via `determine_crop_region()`
- Local (cropped) keypoints converted to global coordinates for visualization

### Temporal Action Recognition
- `TemporalActionRecognizer` class buffers 30 frames of keypoint history
- Computes velocities/accelerations via `_get_keypoint_velocity()` 
- Detects: jumping (upward ankle motion), running/walking (horizontal body velocity), fighting (rapid wrist motion), falling (downward acceleration + lying stillness)
- **Scale-aware thresholds**: All velocity thresholds scale with person's torso size via `_get_pose_scale()`
- **Action smoothing**: Uses vote-based stabilization in `_stabilize_action()` to prevent jitter

## Critical Conventions

### Coordinate System
- **Keypoints format**: `[y, x, confidence]` where y/x are **normalized (0-1)**
- **y=0 is top**, y=1 is bottom (inverted from typical image coordinates)
- Example: `keypoints[KEYPOINT_NOSE]` returns `[0.25, 0.50, 0.95]` for nose at 25% from top, 50% from left

### Edge TPU Specifics
- **Quantized models** (.tflite with `_edgetpu` suffix): Use int8 inference with scale/zero-point calibration
- **Input preprocessing**: Convert to int8 via `(data / input_scale) + input_zero_point`, clamp to [-128, 127]
- **Output dequantization**: `(output - output_zero_point) * output_scale`
- **Device assignment**: Use `usb:0`, `usb:1` for dual TPU setup (see `_make_on_first_available()`)
- **Thread safety**: Each TPU interpreter must run in its own thread; no concurrent `.invoke()` calls

### Threading & Synchronization
- `frame_lock`/`frame_cond`: Protects `latest_frame` and `frame_seq` for producer-consumer pattern
- Inference threads wait on `frame_cond.wait()` and check `frame_seq` to avoid reprocessing same frame
- Copy frames before releasing lock: `frame = latest_frame.copy()` prevents race conditions

## Key Files & Responsibilities

- [server.py](server.py): Main entry point, Flask routes (`/`, `/video_feed`, `/stats`), thread orchestration
- [object_detector.py](object_detector.py): YOLO inference encapsulation with NMS postprocessing (`_non_max_suppression()`)
- [pose_estimator.py](pose_estimator.py): Static pose classification (`estimate_pose()`) + temporal action detection (`TemporalActionRecognizer`)
- [cropping_algorithm.py](cropping_algorithm.py): MoveNet adaptive crop logic (from Google's pose estimation examples)
- [pycoral_examples/](pycoral_examples/): Reference implementations (not used in production, only for learning)
- [audio_classifier.py](audio_classifier.py): Planned audio analysis (crying detection) - will run on Pi's CPU alongside TPU inference

## Future Development Roadmap

### Custom Hazard Detection Model (Planned)
- Replace current YOLO+COCO with custom object detection model
- Target classes: knives, fire, firearms, electrical outlets, cleaning chemicals, stairs/windows (proximity alerts)
- Will require training pipeline for custom .tflite model with Edge TPU quantization

### Audio Classification (Planned)
- Implement [audio_classifier.py](audio_classifier.py) to detect crying, screaming, glass breaking
- Run on Raspberry Pi CPU in separate thread to avoid TPU contention
- Consider librosa/tensorflow-lite audio models optimized for ARM

### Mobile App Integration (External React Native App)
- **WebSocket endpoint**: Real-time event notifications (fall detected, hazard identified, etc.)
  - Will need to add `flask-socketio` or similar for bi-directional communication
  - Event schema: `{type: 'alert', action: 'Fall Detected', confidence: 0.95, timestamp: ...}`
- **MJPEG endpoint**: Live video streaming to mobile app 
  - Mobile app developed separately but will consume these endpoints

## Development Workflows

### Activation of Virtual Environment
```bash
source venv/bin/activate
```

### Running the Server
```bash
python3 server.py \
  --yolo_model models/object/yolo11n_edgetpu.tflite \
  --movenet_model models/pose/movenet_single_pose_lightning_ptq_edgetpu.tflite \
  --labels models/object/coco_labels.txt \
  --camera_idx 0
```
Access web UI at `http://localhost:5000` for live video + stats dashboard.

### Testing Edge TPU Availability
```bash
# Check Edge TPU runtime library
ldconfig -p | grep edgetpu

# List connected TPUs
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

### Adding New Actions to Temporal Recognition
1. Implement detection in `_classify_action()` using velocity/displacement helpers
2. Add color mapping in `get_action_color()` for UI display
3. Consider scale factor: `effective_scale = scale_factor / 0.3` for threshold adjustment
4. Add to `dynamic_actions` or `static_actions` set in `_stabilize_action()` for smoothing
5. For hazard-related actions (e.g., reaching for socket/knife), combine pose data with object detection proximity

### WebSocket Notifications 
```python
# Example pattern for Flask-SocketIO integration
from flask_socketio import SocketIO, emit

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    emit('status', {'msg': 'Connected to child monitor'})

# In inference threads, emit alerts:
if movenet_action_label == "Fall Detected":
    socketio.emit('alert', {
        'type': 'fall',
        'confidence': 0.95,
        'timestamp': time.time()
    })
```

### Modifying Quantization Handling
- Update preprocessing in [object_detector.py](object_detector.py#L100) or [server.py](server.py#L140) 
- Always copy quantization params: `details.get('quantization', (1.0, 0))` returns `(scale, zero_point)`
- Test with both quantized and float models by checking `input_dtype`

## Common Pitfalls

1. **Forgetting to copy frames**: Always `frame.copy()` after acquiring lock to prevent shared memory corruption
2. **Mixing local/global keypoints**: Pose estimation uses local (cropped) coordinates; visualization requires global conversion
3. **Ignoring scale factor**: Action thresholds must scale with person size or they fail for distant/close subjects
4. **Blocking inference threads**: Never call `cv2.imshow()` or long operations inside inference threads (use Flask routes instead)
5. **Incorrect TPU device strings**: Use `usb:0` not `usb` or `/dev/bus/usb/...`
6. **Raspberry Pi resource limits**: Pi 5 has 8GB RAM; monitor memory usage when buffering frames/audio
7. **USB bandwidth**: 2 TPUs + webcam share USB bus; use USB 3.0 ports for TPUs, test different port configurations if performance issues arise

## External Dependencies

**Current**:
- **PyCoral**: Google's Edge TPU Python API (`pycoral.adapters`, `pycoral.utils.edgetpu`)
- **Flask**: Web server for video streaming (multipart MJPEG via `Response(mimetype='multipart/x-mixed-replace')`)
- **OpenCV (cv2)**: Camera capture, image preprocessing, drawing overlays
- **numpy**: Array operations for keypoint math
- **sounddevice/tflite**: Audio processing for crying detection (CPU-based inference)
- **Flask-SocketIO**: WebSocket support for mobile app notifications

No training code exists here - this is inference-only. Models are pre-quantized `.tflite` files. Custom hazard detection model training will be a separate pipeline.
