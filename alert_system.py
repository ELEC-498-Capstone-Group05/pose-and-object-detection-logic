import time
import logging
import numpy as np
import cv2
from collections import deque
from flask_socketio import SocketIO, emit

# Configure logging
logger = logging.getLogger(__name__)

# Keypoint indices (MoveNet)
KEYPOINT_NOSE = 0
KEYPOINT_LEFT_WRIST = 9
KEYPOINT_RIGHT_WRIST = 10
KEYPOINT_LEFT_ANKLE = 15
KEYPOINT_RIGHT_ANKLE = 16

class AlertSystem:
    def __init__(self, app):
        """
        Initialize the Alert System.
        
        Args:
            app: The Flask application instance
        """
        self.app = app
        self.socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
        self.last_alert_times = {}
        self.cooldown = 3.0  # Minimum seconds between repeat alerts
        self.default_fps = 30.0
        self.safezone_config = {
            'enabled': False,
            'zone': {
                'x': 0.1,
                'y': 0.1,
                'width': 0.8,
                'height': 0.8,
            }
        }
        self.safezone_state = {
            'outside': False,
            'has_person': False,
        }
        self.safezone_min_person_score = 0.45
        self.safezone_min_box_size = 0.03
        self.safezone_debug_interval = 15
        self._safezone_frame_counter = 0
        
        # Alert Configuration
        self.alerts_config = {
            "fighting": {
                "name": "Fighting Detected",
                "type": "action",
                "trigger": "Fighting",
                "message": "Fighting detected!",
                "severity": "medium",
                "enabled": True
            },
            "fall": {
                "name": "Fall Detected",
                "type": "action",
                "trigger": "Fall Detected",
                "message": "Fall detected!",
                "severity": "low",
                "enabled": True
            },
            "fall_no_recovery": {
                "name": "Fall + No Recovery",
                "type": "temporal_action",
                "message": "Fall detected with no recovery!",
                "severity": "high",
                "enabled": True,
                "duration_s": 8.0,
                "cooldown_s": 8.0
            },
            "head_impact_suspected": {
                "name": "Head Impact Suspected",
                "type": "action",
                "trigger": "Head Impact Suspected",
                "message": "Head impact suspected!",
                "severity": "high",
                "enabled": True,
                "cooldown_s": 5.0
            },
             "knife": {
                "name": "Knife Detected",
                "type": "object",
                "class_id": 43, # COCO Class ID for Knife
                "message": "Knife detected!",
                "severity": "high",
                "enabled": True
            },
            "weapon": {
                "name": "Weapon Detected",
                "type": "object",
                "class_id": 34, # COCO Class ID for Baseball Bat (as a proxy for weapon)
                "message": "Weapon detected!",
                "severity": "high",
                "enabled": True
            },
            "knife_in_hand": {
                "name": "Knife in Hand",
                "type": "spatial_proximity",
                "object_class_id": 43, # Knife
                "keypoints": [KEYPOINT_LEFT_WRIST, KEYPOINT_RIGHT_WRIST],
                "threshold": 0.1, # Normalized distance threshold
                "message": "Knife in hand detected!",
                "severity": "high",
                "enabled": True
            },
            "jumping_on_couch": {
                "name": "Jumping on Couch",
                "type": "spatial_overlap",
                "action": "Jumping",
                "object_class_id": 57, # COCO Class ID for Couch/Sofa
                "message": "Jumping on furniture detected!",
                "severity": "low",
                "enabled": True
            },
            "climbing_hazard": {
                "name": "Climbing Hazard",
                "type": "climbing",
                "message": "Climbing hazard detected!",
                "severity": "low",
                "enabled": True,
                "window_s": 1.5,
                "min_upward_delta": 0.12,
                "min_samples": 6,
                "require_furniture_overlap": True,
                "furniture_class_ids": [56, 57, 59],
                "keypoint_confidence": 0.35,
                "cooldown_s": 6.0
            },
            "screaming": {
                "name": "Screaming Detected",
                "type": "audio",
                "trigger_classes": [11], # Screaming
                "message": "Screaming detected!",
                "severity": "medium",
                "enabled": True
            },
            "crying": {
                "name": "Crying Detected",
                "type": "audio",
                "trigger_classes": [19, 20], # Crying, sobbing; Baby cry
                "message": "Crying detected!",
                "severity": "medium",
                "enabled": True
            },
            "shatter": {
                "name": "Shatter Detected",
                "type": "audio",
                "trigger_classes": [435, 437, 463, 464], # Glass, Shatter, Smash, Breaking
                "message": "Shatter/Breaking sound detected!",
                "severity": "medium",
                "enabled": True
            },
            "choking_cough_distress": {
                "name": "Choking/Cough Distress",
                "type": "audio_pattern",
                "trigger_classes": [42, 43],  # Cough, Throat clearing
                "message": "Repeated coughing distress pattern detected!",
                "severity": "high",
                "enabled": True,
                "window_s": 4.0,
                "min_count": 3,
                "min_confidence": 0.35,
                "cooldown_s": 6.0
            },
            "unusual_silence": {
                "name": "Unusual Silence",
                "type": "silence",
                "message": "Unusual silence detected.",
                "severity": "medium",
                "enabled": True,
                "silence_duration_s": 20.0,
                "silent_db_threshold": -55.0,
                "motion_grace_s": 3.0,
                "person_required_within_s": 10.0,
                "cooldown_s": 12.0
            },
            "safezone_exit": {
                "name": "Safezone Exit",
                "type": "safezone",
                "message": "Person left safezone!",
                "severity": "high",
                "enabled": True
            },
            "monitoring_failure": {
                "name": "Monitoring Failure",
                "type": "monitoring",
                "message": "Monitoring failure detected.",
                "severity": "high",
                "enabled": True,
                "stale_frames_threshold": 45,
                "obstruction_persist_frames": 45,
                "dark_mean_threshold": 28.0,
                "blur_var_threshold": 20.0,
                "camera_failure_threshold": 8,
                "cooldown_s": 8.0
            }
        }

        self.alert_status = {
            key: {
                'enabled': bool(cfg.get('enabled', False)),
                'last_trigger_unix_ms': 0,
                'trigger_count': 0,
                'suppressed_count': 0,
                'cooldown_remaining_ms': 0,
                'last_message': None,
                'last_reason': None,
            }
            for key, cfg in self.alerts_config.items()
        }

        self.temporal_state = {
            'fall_no_recovery': {
                'active': False,
                'start_ts': None,
                'alerted': False,
            },
            'climbing': {
                'nose_history': deque(maxlen=120),
            },
            'audio': {
                'cough_events': deque(maxlen=32),
                'silence_start_ts': None,
                'silence_latched': False,
            },
            'activity': {
                'last_motion_ts': 0.0,
                'last_person_seen_ts': 0.0,
            },
            'monitoring': {
                'stale_counter': 0,
                'visual_counter': 0,
                'last_visual_score': {'mean_luma': None, 'lap_var': None},
            },
        }
        
        # Setup socket handlers
        self._setup_socket_handlers()

    def _setup_socket_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to alert system")
            emit('status', {'msg': 'Connected to Child Monitor Alert System'})

    def _trigger(self, alert_key, message, cooldown_override=None, metadata=None):
        """Emit alert if cooldown has passed."""
        now = time.time()
        last_time = self.last_alert_times.get(alert_key, 0)

        status = self.alert_status.setdefault(alert_key, {
            'enabled': True,
            'last_trigger_unix_ms': 0,
            'trigger_count': 0,
            'suppressed_count': 0,
            'cooldown_remaining_ms': 0,
            'last_message': None,
            'last_reason': None,
        })

        cfg = self.alerts_config.get(alert_key, {})
        cooldown_s = (
            float(cooldown_override)
            if cooldown_override is not None
            else float(cfg.get('cooldown_s', self.cooldown))
        )

        if now - last_time > cooldown_s:
            logger.info(f"ALERT TRIGGERED: {message}")
            severity = str(cfg.get('severity', 'medium'))
            payload = {
                'type': alert_key,
                'message': message,
                'severity': severity,
                'timestamp': now * 1000  # Convert to milliseconds
            }
            if isinstance(metadata, dict) and metadata:
                payload['metadata'] = metadata

            self.socketio.emit('alert', payload)
            self.last_alert_times[alert_key] = now
            status['last_trigger_unix_ms'] = int(now * 1000)
            status['trigger_count'] += 1
            status['cooldown_remaining_ms'] = int(cooldown_s * 1000)
            status['last_message'] = message
            if isinstance(metadata, dict):
                status['last_reason'] = metadata.get('reason')
            return True

        status['suppressed_count'] += 1
        status['cooldown_remaining_ms'] = int(max(0.0, cooldown_s - (now - last_time)) * 1000)
        logger.debug(
            "Alert suppressed by cooldown: key=%s remaining=%.2fs",
            alert_key,
            max(0.0, cooldown_s - (now - last_time))
        )
        return False

    def get_alert_status(self):
        now = time.time()
        snapshot = {}
        for key, cfg in self.alerts_config.items():
            status = self.alert_status.setdefault(key, {
                'enabled': bool(cfg.get('enabled', False)),
                'last_trigger_unix_ms': 0,
                'trigger_count': 0,
                'suppressed_count': 0,
                'cooldown_remaining_ms': 0,
                'last_message': None,
                'last_reason': None,
            })
            last_time = self.last_alert_times.get(key, 0.0)
            cooldown_s = float(cfg.get('cooldown_s', self.cooldown))
            cooldown_remaining_ms = int(max(0.0, cooldown_s - (now - last_time)) * 1000)
            snapshot[key] = {
                **status,
                'enabled': bool(cfg.get('enabled', False)),
                'cooldown_remaining_ms': cooldown_remaining_ms,
            }
        return snapshot

    def _update_presence_and_motion_state(self, has_person, action_label):
        now = time.time()
        activity_state = self.temporal_state['activity']
        if has_person:
            activity_state['last_person_seen_ts'] = now

        motion_actions = {
            'Jumping', 'Running', 'Walking', 'Fighting', 'Punching',
            'Kicking', 'Head Impact Suspected'
        }
        if action_label in motion_actions:
            activity_state['last_motion_ts'] = now

    def _process_fall_no_recovery(self, action_label):
        cfg = self.alerts_config.get('fall_no_recovery', {})
        if not cfg.get('enabled', False):
            return

        now = time.time()
        state = self.temporal_state['fall_no_recovery']
        duration_s = float(cfg.get('duration_s', 8.0))

        if action_label == 'Fall Detected':
            if not state['active']:
                state['active'] = True
                state['start_ts'] = now
                state['alerted'] = False

            elapsed = now - (state['start_ts'] or now)
            if (not state['alerted']) and elapsed >= duration_s:
                state['alerted'] = self._trigger(
                    'fall_no_recovery',
                    cfg['message'],
                    metadata={'reason': 'no_recovery_timeout', 'elapsed_s': round(elapsed, 2)}
                )
        else:
            state['active'] = False
            state['start_ts'] = None
            state['alerted'] = False

    def _is_keypoint_in_box(self, kp_x, kp_y, box):
        x_min = box[0] - box[2] / 2.0
        x_max = box[0] + box[2] / 2.0
        y_min = box[1] - box[3] / 2.0
        y_max = box[1] + box[3] / 2.0
        return x_min <= kp_x <= x_max and y_min <= kp_y <= y_max

    def _process_climbing_hazard(self, action_label, norm_boxes, class_map, movenet_results):
        cfg = self.alerts_config.get('climbing_hazard', {})
        if not cfg.get('enabled', False) or movenet_results is None:
            return

        nose = movenet_results[KEYPOINT_NOSE]
        if float(nose[2]) < float(cfg.get('keypoint_confidence', 0.35)):
            return

        now = time.time()
        history = self.temporal_state['climbing']['nose_history']
        history.append((now, float(nose[0]), float(nose[1])))

        window_s = float(cfg.get('window_s', 1.5))
        while history and (now - history[0][0]) > window_s:
            history.popleft()

        min_samples = int(cfg.get('min_samples', 6))
        if len(history) < min_samples:
            return

        y_start = history[0][1]
        y_end = history[-1][1]
        upward_delta = y_start - y_end
        if upward_delta < float(cfg.get('min_upward_delta', 0.12)):
            return

        furniture_overlap = False
        furniture_ids = cfg.get('furniture_class_ids', [56, 57, 59])
        for furniture_id in furniture_ids:
            for box_idx in class_map.get(furniture_id, []):
                if self._is_keypoint_in_box(float(nose[1]), float(nose[0]), norm_boxes[box_idx]):
                    furniture_overlap = True
                    break
            if furniture_overlap:
                break

        if cfg.get('require_furniture_overlap', True) and not furniture_overlap:
            return

        if action_label == 'Jumping':
            return

        self._trigger(
            'climbing_hazard',
            cfg['message'],
            metadata={
                'reason': 'sustained_upward_motion',
                'upward_delta': round(float(upward_delta), 3),
                'samples': len(history),
                'furniture_overlap': bool(furniture_overlap),
            },
        )

    def _is_visual_obstruction(self, frame, dark_mean_threshold, blur_var_threshold):
        if frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.temporal_state['monitoring']['last_visual_score'] = {
            'mean_luma': round(mean_luma, 3),
            'lap_var': round(lap_var, 3),
        }
        return mean_luma <= dark_mean_threshold or lap_var <= blur_var_threshold

    def _process_monitoring_failure(self, frame, monitoring_context):
        cfg = self.alerts_config.get('monitoring_failure', {})
        if not cfg.get('enabled', False):
            return

        context = monitoring_context if isinstance(monitoring_context, dict) else {}
        state = self.temporal_state['monitoring']

        yolo_age = int(context.get('yolo_result_age_frames', 0) or 0)
        movenet_age = int(context.get('movenet_result_age_frames', 0) or 0)
        camera_read_failures = int(context.get('camera_read_failures', 0) or 0)

        stale_threshold = int(cfg.get('stale_frames_threshold', 45))
        stale_now = yolo_age >= stale_threshold or movenet_age >= stale_threshold
        state['stale_counter'] = state['stale_counter'] + 1 if stale_now else 0

        visual_now = self._is_visual_obstruction(
            frame,
            dark_mean_threshold=float(cfg.get('dark_mean_threshold', 28.0)),
            blur_var_threshold=float(cfg.get('blur_var_threshold', 20.0)),
        )
        state['visual_counter'] = state['visual_counter'] + 1 if visual_now else 0

        reasons = []
        if state['stale_counter'] >= int(cfg.get('obstruction_persist_frames', 45)):
            reasons.append('stale_inference')
        if state['visual_counter'] >= int(cfg.get('obstruction_persist_frames', 45)):
            reasons.append('visual_obstruction')
        if camera_read_failures >= int(cfg.get('camera_failure_threshold', 8)):
            reasons.append('camera_read_failures')

        if not reasons:
            return

        self._trigger(
            'monitoring_failure',
            cfg['message'],
            metadata={
                'reason': ','.join(reasons),
                'yolo_result_age_frames': yolo_age,
                'movenet_result_age_frames': movenet_age,
                'camera_read_failures': camera_read_failures,
                'visual': state['last_visual_score'],
            },
        )

    def _process_cough_distress(self, detections, timestamp):
        cfg = self.alerts_config.get('choking_cough_distress', {})
        if not cfg.get('enabled', False):
            return

        now = float(timestamp if timestamp is not None else time.time())
        min_conf = float(cfg.get('min_confidence', 0.35))
        trigger_classes = cfg.get('trigger_classes', [42, 43])
        cough_events = self.temporal_state['audio']['cough_events']

        detected = False
        for class_id in trigger_classes:
            if class_id in detections and float(detections[class_id]['score']) >= min_conf:
                cough_events.append((now, class_id, float(detections[class_id]['score'])))
                detected = True

        if not detected:
            return

        window_s = float(cfg.get('window_s', 4.0))
        while cough_events and (now - cough_events[0][0]) > window_s:
            cough_events.popleft()

        min_count = int(cfg.get('min_count', 3))
        if len(cough_events) >= min_count:
            avg_conf = float(np.mean([ev[2] for ev in cough_events]))
            self._trigger(
                'choking_cough_distress',
                f"{cfg['message']} ({int(avg_conf * 100)}%)",
                metadata={
                    'reason': 'cough_burst_pattern',
                    'count': len(cough_events),
                    'window_s': window_s,
                    'avg_confidence': round(avg_conf, 3),
                },
            )

    def _process_unusual_silence(self, detections, audio_db, timestamp):
        cfg = self.alerts_config.get('unusual_silence', {})
        if not cfg.get('enabled', False):
            return

        now = float(timestamp if timestamp is not None else time.time())
        audio_state = self.temporal_state['audio']
        activity_state = self.temporal_state['activity']

        person_recent_window = float(cfg.get('person_required_within_s', 10.0))
        person_recent = (now - activity_state['last_person_seen_ts']) <= person_recent_window
        if not person_recent:
            audio_state['silence_start_ts'] = None
            audio_state['silence_latched'] = False
            return

        motion_grace_s = float(cfg.get('motion_grace_s', 3.0))
        no_motion = (now - activity_state['last_motion_ts']) >= motion_grace_s

        silence_class_present = 494 in detections and float(detections[494]['score']) >= 0.4
        db_is_silent = audio_db is not None and float(audio_db) <= float(cfg.get('silent_db_threshold', -55.0))
        is_silent = silence_class_present or db_is_silent

        if no_motion and is_silent:
            if audio_state['silence_start_ts'] is None:
                audio_state['silence_start_ts'] = now

            silence_duration = now - audio_state['silence_start_ts']
            if (
                silence_duration >= float(cfg.get('silence_duration_s', 20.0))
                and not audio_state['silence_latched']
            ):
                triggered = self._trigger(
                    'unusual_silence',
                    f"{cfg['message']} ({int(silence_duration)}s)",
                    metadata={
                        'reason': 'fixed_timeout_silence',
                        'silence_duration_s': round(float(silence_duration), 2),
                        'audio_db': float(audio_db) if audio_db is not None else None,
                    },
                )
                if triggered:
                    audio_state['silence_latched'] = True
        else:
            audio_state['silence_start_ts'] = None
            audio_state['silence_latched'] = False

    def get_safezone_config(self):
        zone = self.safezone_config['zone']
        return {
            'enabled': bool(self.safezone_config['enabled']),
            'zone': {
                'x': float(zone['x']),
                'y': float(zone['y']),
                'width': float(zone['width']),
                'height': float(zone['height'])
            }
        }

    def _validate_safezone_payload(self, payload):
        errors = {}

        if not isinstance(payload, dict):
            return None, {'payload': 'Expected JSON object payload.'}

        if 'enabled' not in payload:
            errors['enabled'] = 'Field is required.'
        elif not isinstance(payload['enabled'], bool):
            errors['enabled'] = 'Must be boolean.'

        zone = payload.get('zone')
        if not isinstance(zone, dict):
            errors['zone'] = 'Field is required and must be an object.'
            return None, errors

        normalized_zone = {}
        for field in ['x', 'y', 'width', 'height']:
            value = zone.get(field)
            if value is None:
                errors[field] = 'Field is required.'
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                errors[field] = 'Must be numeric.'
                continue

            if numeric_value < 0.0 or numeric_value > 1.0:
                errors[field] = 'Must be in range [0, 1].'
                continue

            normalized_zone[field] = numeric_value

        if 'x' in normalized_zone and 'width' in normalized_zone:
            if normalized_zone['x'] + normalized_zone['width'] > 1.0:
                errors['zone'] = 'x + width must be <= 1.0.'

        if 'y' in normalized_zone and 'height' in normalized_zone:
            if normalized_zone['y'] + normalized_zone['height'] > 1.0:
                errors['zone'] = 'y + height must be <= 1.0.'

        if errors:
            return None, errors

        normalized = {
            'enabled': payload['enabled'],
            'zone': normalized_zone
        }
        return normalized, None

    def update_safezone_config(self, payload):
        normalized, errors = self._validate_safezone_payload(payload)
        if errors is not None:
            logger.warning("Safezone config rejected: errors=%s payload=%s", errors, payload)
            return None, errors

        self.safezone_config = normalized
        logger.info("Safezone config updated: %s", self.safezone_config)
        if not normalized['enabled']:
            self.safezone_state = {'outside': False, 'has_person': False}
        return self.get_safezone_config(), None

    def _normalize_boxes_to_frame(self, boxes, input_size, meta):
        if len(boxes) == 0:
            return np.array([])

        boxes = np.asarray(boxes, dtype=np.float32)
        input_w, input_h = input_size

        boxes_px = boxes.copy()
        if float(np.max(np.abs(boxes_px))) <= 2.0:
            boxes_px[:, 0] *= float(input_w)
            boxes_px[:, 1] *= float(input_h)
            boxes_px[:, 2] *= float(input_w)
            boxes_px[:, 3] *= float(input_h)

        if not isinstance(meta, dict) or not meta.get('letterbox'):
            return boxes_px / np.array([input_w, input_h, input_w, input_h], dtype=np.float32)

        scale = float(meta.get('scale', 1.0))
        pad_left = float(meta.get('pad_left', 0.0))
        pad_top = float(meta.get('pad_top', 0.0))
        frame_w = float(meta.get('frame_width', 0.0))
        frame_h = float(meta.get('frame_height', 0.0))

        if scale <= 0.0 or frame_w <= 0.0 or frame_h <= 0.0:
            return boxes_px / np.array([input_w, input_h, input_w, input_h], dtype=np.float32)

        norm_boxes = np.zeros_like(boxes_px, dtype=np.float32)
        norm_boxes[:, 0] = ((boxes_px[:, 0] - pad_left) / scale) / frame_w
        norm_boxes[:, 1] = ((boxes_px[:, 1] - pad_top) / scale) / frame_h
        norm_boxes[:, 2] = (boxes_px[:, 2] / scale) / frame_w
        norm_boxes[:, 3] = (boxes_px[:, 3] / scale) / frame_h
        return np.clip(norm_boxes, 0.0, 1.0)

    def _safezone_person_status(self, norm_boxes, class_ids, scores):
        if len(norm_boxes) == 0 or len(class_ids) == 0:
            return False, False, [], []

        zone = self.safezone_config['zone']
        zx_min = zone['x']
        zy_min = zone['y']
        zx_max = zone['x'] + zone['width']
        zy_max = zone['y'] + zone['height']

        has_person = False
        person_centers = []
        person_debug = []

        for i, cid in enumerate(class_ids):
            if int(cid) != 0:
                continue

            score = float(scores[i]) if len(scores) > i else 0.0
            if score < self.safezone_min_person_score:
                continue

            box = norm_boxes[i]
            px = float(box[0])
            py = float(box[1])
            bw = float(box[2])
            bh = float(box[3])
            person_debug.append({
                'center': (round(px, 3), round(py, 3)),
                'score': round(score, 3),
                'size': (round(bw, 3), round(bh, 3)),
            })

            if bw < self.safezone_min_box_size or bh < self.safezone_min_box_size:
                continue

            has_person = True
            person_centers.append((round(px, 3), round(py, 3)))

            if not (zx_min <= px <= zx_max and zy_min <= py <= zy_max):
                return True, True, person_centers, person_debug

        return has_person, False, person_centers, person_debug

    def _should_trigger_safezone_exit(self, has_person, outside_now):
        prev_outside = self.safezone_state['outside']

        if not has_person:
            self.safezone_state['has_person'] = False
            return False

        self.safezone_state['has_person'] = True
        self.safezone_state['outside'] = outside_now

        if outside_now and not prev_outside:
            return True

        return False

    def _log_safezone_debug(self, has_person, outside_now, should_trigger, person_centers, person_debug):
        self._safezone_frame_counter += 1
        if should_trigger or self._safezone_frame_counter % self.safezone_debug_interval == 0:
            zone = self.safezone_config['zone']
            logger.warning(
                "SAFEZONE DEBUG: enabled=%s has_person=%s outside_now=%s trigger=%s prev_state=%s centers=%s person_debug=%s zone=%s",
                self.safezone_config.get('enabled', False),
                has_person,
                outside_now,
                should_trigger,
                self.safezone_state,
                person_centers,
                person_debug,
                {
                    'x': round(zone['x'], 3),
                    'y': round(zone['y'], 3),
                    'width': round(zone['width'], 3),
                    'height': round(zone['height'], 3),
                }
            )

    def process_frame(self, yolo_results, movenet_results, action_label, input_size, frame=None, monitoring_context=None):
        """
        Process frame results and trigger alerts.
        
        Args:
            yolo_results: Tuple (boxes, class_ids, scores, meta) from YOLO
            movenet_results: Numpy array (17, 3) of keypoints [y, x, conf] (normalized)
            action_label: Current detected action string
            input_size: Tuple (width, height) of YOLO input for normalizing boxes
        """
        if yolo_results is None:
            boxes, class_ids, scores, meta = [], [], [], None
        else:
            boxes, class_ids, scores, meta = yolo_results

        norm_boxes = self._normalize_boxes_to_frame(boxes, input_size, meta)

        # 1. Check Action Alerts
        for key, config in self.alerts_config.items():
            if not config['enabled']: continue
            
            if config['type'] == 'action':
                if action_label == config['trigger']:
                    self._trigger(key, config['message'])

        has_person = False
        for i, cid in enumerate(class_ids):
            if int(cid) != 0:
                continue
            score = float(scores[i]) if len(scores) > i else 0.0
            if score >= self.safezone_min_person_score:
                has_person = True
                break
        self._update_presence_and_motion_state(has_person, action_label)
        self._process_fall_no_recovery(action_label)

        # 2. Check Object and Spatial Alerts
        # Map class_id to indices in current results
        class_map = {}
        for i, cid in enumerate(class_ids):
            if cid not in class_map: class_map[cid] = []
            class_map[cid].append(i)

        for key, config in self.alerts_config.items():
            if not config['enabled']: continue

            # Simple Object Detection
            if config['type'] == 'object':
                if config['class_id'] in class_map:
                    self._trigger(key, config['message'])

            # Knife in Hand (Proximity)
            elif config['type'] == 'spatial_proximity' and movenet_results is not None:
                target_cid = config['object_class_id']
                if target_cid in class_map:
                    # Check each instance of the object
                    for box_idx in class_map[target_cid]:
                        box = norm_boxes[box_idx] # [cx, cy, w, h] normalized
                        obj_center = np.array([box[0], box[1]]) # [x, y]
                        
                        # Check against specified keypoints
                        for kp_idx in config['keypoints']:
                            kp = movenet_results[kp_idx] # [y, x, conf]
                            if kp[2] < 0.3: continue # Low confidence
                            
                            kp_pos = np.array([kp[1], kp[0]]) # Swap to [x, y]
                            dist = np.linalg.norm(obj_center - kp_pos)
                            
                            if dist < config['threshold']:
                                self._trigger(key, config['message'])
                                break

            # Jumping on Couch (Overlap)
            elif config['type'] == 'spatial_overlap' and movenet_results is not None:
                if action_label == config['action']:
                    target_cid = config['object_class_id']
                    if target_cid in class_map:
                        for box_idx in class_map[target_cid]:
                            box = norm_boxes[box_idx] # [cx, cy, w, h]
                            # Define object bounds
                            x_min = box[0] - box[2]/2
                            x_max = box[0] + box[2]/2
                            y_min = box[1] - box[3]/2
                            y_max = box[1] + box[3]/2
                            
                            # Check if ankles are inside the box
                            # Ankle keypoints: 15 (Left), 16 (Right)
                            for kp_idx in [KEYPOINT_LEFT_ANKLE, KEYPOINT_RIGHT_ANKLE]:
                                kp = movenet_results[kp_idx] # [y, x, conf]
                                if kp[2] < 0.3: continue
                                
                                kx, ky = kp[1], kp[0]
                                if x_min <= kx <= x_max and y_min <= ky <= y_max:
                                    self._trigger(key, config['message'])
                                    break

        self._process_climbing_hazard(action_label, norm_boxes, class_map, movenet_results)

        safezone_alert = self.alerts_config.get('safezone_exit')
        safezone_enabled = self.safezone_config.get('enabled', False)
        if not safezone_enabled:
            self.safezone_state = {'outside': False, 'has_person': False}
        elif safezone_alert and safezone_alert.get('enabled', False):
            has_person, outside_now, person_centers, person_debug = self._safezone_person_status(norm_boxes, class_ids, scores)
            should_trigger = self._should_trigger_safezone_exit(has_person, outside_now)
            # self._log_safezone_debug(has_person, outside_now, should_trigger, person_centers, person_debug)
            if should_trigger:
                self._trigger('safezone_exit', safezone_alert['message'])

        self._process_monitoring_failure(frame, monitoring_context)

    def process_audio(self, detections, audio_db=None, timestamp=None):
        """
        Process audio classifier results
        Args:
            detections: Dict {class_idx: {'score': float, 'name': str}}
        """
        if detections is None:
            detections = {}

        self._process_cough_distress(detections, timestamp)
        self._process_unusual_silence(detections, audio_db, timestamp)

        for key, config in self.alerts_config.items():
            if not config['enabled'] or config['type'] != 'audio':
                continue
                
            for class_id in config['trigger_classes']:
                if class_id in detections:
                    # Found a matching class
                    confidence = detections[class_id]['score']
                    msg = f"{config['message']} ({int(confidence*100)}%)"
                    self._trigger(key, msg)
                    break
