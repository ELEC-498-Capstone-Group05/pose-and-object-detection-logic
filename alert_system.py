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
        
        # Alert Configuration
        self.alerts_config = {
            "fighting": {
                "name": "Fighting Detected",
                "type": "action",
                "trigger": "Fighting",
                "message": "Fighting detected!",
                "enabled": True
            },
            "fall": {
                "name": "Fall Detected",
                "type": "action",
                "trigger": "Fall Detected",
                "message": "Fall detected!",
                "enabled": True
            },
             "knife": {
                "name": "Knife Detected",
                "type": "object",
                "class_id": 43, # COCO Class ID for Knife
                "message": "Knife detected!",
                "enabled": True
            },
            "weapon": {
                "name": "Weapon Detected",
                "type": "object",
                "class_id": 34, # COCO Class ID for Baseball Bat (as a proxy for weapon)
                "message": "Weapon detected!",
                "enabled": True
            },
            "knife_in_hand": {
                "name": "Knife in Hand",
                "type": "spatial_proximity",
                "object_class_id": 43, # Knife
                "keypoints": [KEYPOINT_LEFT_WRIST, KEYPOINT_RIGHT_WRIST],
                "threshold": 0.1, # Normalized distance threshold
                "message": "Knife in hand detected!",
                "enabled": True
            },
            "jumping_on_couch": {
                "name": "Jumping on Couch",
                "type": "spatial_overlap",
                "action": "Jumping",
                "object_class_id": 57, # COCO Class ID for Couch/Sofa
                "message": "Jumping on furniture detected!",
                "enabled": True
            },
            "screaming": {
                "name": "Screaming Detected",
                "type": "audio",
                "trigger_classes": [11], # Screaming
                "message": "Screaming detected!",
                "enabled": True
            },
            "crying": {
                "name": "Crying Detected",
                "type": "audio",
                "trigger_classes": [19, 20], # Crying, sobbing; Baby cry
                "message": "Crying detected!",
                "enabled": True
            },
            "shatter": {
                "name": "Shatter Detected",
                "type": "audio",
                "trigger_classes": [435, 437, 463, 464], # Glass, Shatter, Smash, Breaking
                "message": "Shatter/Breaking sound detected!",
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
        return False

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
            boxes, class_ids, scores = [], [], []
        else:
            boxes, class_ids, scores, _ = yolo_results
            
        # Normalize boxes for spatial checks: [cx, cy, w, h] -> normalized [cx, cy, w, h]
        input_w, input_h = input_size
        norm_boxes = []
        if len(boxes) > 0:
            norm_boxes = boxes / np.array([input_w, input_h, input_w, input_h])

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
