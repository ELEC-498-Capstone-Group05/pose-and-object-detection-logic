import time
import logging
import numpy as np
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
                "severity": "high",
                "enabled": True
            },
            "fall": {
                "name": "Fall Detected",
                "type": "action",
                "trigger": "Fall Detected",
                "message": "Fall detected!",
                "severity": "high",
                "enabled": True
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
                "severity": "medium",
                "enabled": True
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
            "safezone_exit": {
                "name": "Safezone Exit",
                "type": "safezone",
                "message": "Person left safezone!",
                "severity": "high",
                "enabled": True
            }
        }
        
        # Setup socket handlers
        self._setup_socket_handlers()

    def _setup_socket_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to alert system")
            emit('status', {'msg': 'Connected to Child Monitor Alert System'})

    def _trigger(self, alert_key, message):
        """Emit alert if cooldown has passed."""
        now = time.time()
        last_time = self.last_alert_times.get(alert_key, 0)
        
        if now - last_time > self.cooldown:
            logger.info(f"ALERT TRIGGERED: {message}")
            self.socketio.emit('alert', {
                'type': alert_key,
                'message': message,
                'timestamp': now
            })
            self.last_alert_times[alert_key] = now
            return True
        logger.debug(
            "Alert suppressed by cooldown: key=%s remaining=%.2fs",
            alert_key,
            max(0.0, self.cooldown - (now - last_time))
        )
        return False

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

    def process_frame(self, yolo_results, movenet_results, action_label, input_size):
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

    def process_audio(self, detections):
        """
        Process audio classifier results
        Args:
            detections: Dict {class_idx: {'score': float, 'name': str}}
        """
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
