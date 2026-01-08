"""
Pose Estimator for MoveNet Keypoints

This module analyzes the 17 keypoints from MoveNet to estimate basic human poses
and complex actions over time.

Features:
- Static pose estimation (standing, sitting, lying down)
- Temporal action recognition (jumping, running, walking, fighting)
- Keypoint buffering and velocity/acceleration computation

Keypoint indices:
    0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
    5: Left Shoulder, 6: Right Shoulder
    7: Left Elbow, 8: Right Elbow
    9: Left Wrist, 10: Right Wrist
    11: Left Hip, 12: Right Hip
    13: Left Knee, 14: Right Knee
    15: Left Ankle, 16: Right Ankle

Each keypoint is [y, x, confidence] where y and x are normalized (0-1).
Note: y=0 is top, y=1 is bottom in the coordinate system.
"""

import numpy as np
from collections import deque

# Keypoint indices for major body parts
KEYPOINT_NOSE = 0
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_ELBOW = 7
KEYPOINT_RIGHT_ELBOW = 8
KEYPOINT_LEFT_WRIST = 9
KEYPOINT_RIGHT_WRIST = 10
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12
KEYPOINT_LEFT_KNEE = 13
KEYPOINT_RIGHT_KNEE = 14
KEYPOINT_LEFT_ANKLE = 15
KEYPOINT_RIGHT_ANKLE = 16

# Confidence threshold for considering a keypoint valid
CONFIDENCE_THRESHOLD = 0.3


def estimate_pose(keypoints, min_confidence=CONFIDENCE_THRESHOLD):
    """
    Estimate the pose of a person based on MoveNet keypoints.
    
    Args:
        keypoints: numpy array of shape (17, 3) where each row is [y, x, confidence]
        min_confidence: minimum confidence threshold for valid keypoints
    
    Returns:
        str: Estimated pose label - "Standing", "Sitting", "Lying Down", or "Unknown"
    """
    if keypoints is None or len(keypoints) != 17:
        return "Unknown"
    
    # Extract key body parts with confidence check
    def get_keypoint(index):
        kp = keypoints[index]
        if kp[2] >= min_confidence:
            return kp[:2]  # Return [y, x]
        return None
    
    # Get critical keypoints
    nose = get_keypoint(KEYPOINT_NOSE)
    left_shoulder = get_keypoint(KEYPOINT_LEFT_SHOULDER)
    right_shoulder = get_keypoint(KEYPOINT_RIGHT_SHOULDER)
    left_hip = get_keypoint(KEYPOINT_LEFT_HIP)
    right_hip = get_keypoint(KEYPOINT_RIGHT_HIP)
    left_knee = get_keypoint(KEYPOINT_LEFT_KNEE)
    right_knee = get_keypoint(KEYPOINT_RIGHT_KNEE)
    left_ankle = get_keypoint(KEYPOINT_LEFT_ANKLE)
    right_ankle = get_keypoint(KEYPOINT_RIGHT_ANKLE)
    
    # Calculate average positions for paired keypoints
    shoulder_y = None
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_y = (left_shoulder[0] + right_shoulder[0]) / 2
    elif left_shoulder is not None:
        shoulder_y = left_shoulder[0]
    elif right_shoulder is not None:
        shoulder_y = right_shoulder[0]
    
    hip_y = None
    if left_hip is not None and right_hip is not None:
        hip_y = (left_hip[0] + right_hip[0]) / 2
    elif left_hip is not None:
        hip_y = left_hip[0]
    elif right_hip is not None:
        hip_y = right_hip[0]
    
    knee_y = None
    if left_knee is not None and right_knee is not None:
        knee_y = (left_knee[0] + right_knee[0]) / 2
    elif left_knee is not None:
        knee_y = left_knee[0]
    elif right_knee is not None:
        knee_y = right_knee[0]
    
    ankle_y = None
    if left_ankle is not None and right_ankle is not None:
        ankle_y = (left_ankle[0] + right_ankle[0]) / 2
    elif left_ankle is not None:
        ankle_y = left_ankle[0]
    elif right_ankle is not None:
        ankle_y = right_ankle[0]
    
    # Need at least shoulders and hips for basic pose estimation
    if shoulder_y is None or hip_y is None:
        return "Unknown"
    
    # Calculate torso length (vertical distance from shoulders to hips)
    torso_length = abs(hip_y - shoulder_y)
    
    # Analyze pose based on body geometry
    # Remember: lower y values = higher on screen (top of image)
    
    # Check for lying down: torso is nearly horizontal (small vertical distance)
    if torso_length < 0.15:
        return "Lying Down"
    
    # If hips are significantly higher than shoulders, orientation is ambiguous
    if hip_y < shoulder_y - 0.05:
        return "Unknown"
    
    # Analyze knee position relative to hips
    if knee_y is not None:
        hip_to_knee = knee_y - hip_y  # Positive means knees are below hips
        
        # Standing: knees are significantly below hips, torso is upright
        if hip_to_knee > 0.15 and torso_length > 0.22:
            if ankle_y is not None and ankle_y > knee_y:
                return "Standing"
            return "Standing"

        # Sitting: knees are roughly at same level or above hips
        # (when sitting, knees bend so they appear higher/at similar y-value to hips)
        if hip_to_knee < 0.12 and torso_length > 0.15:
            return "Sitting"
    
    # Default classification based on torso length
    if torso_length > 0.28:
        return "Standing"
    elif torso_length > 0.15:
        return "Sitting"
    else:
        return "Unknown"


def get_pose_color(pose_label):
    """
    Get a color for displaying the pose label.
    
    Args:
        pose_label: The pose label string
    
    Returns:
        tuple: BGR color tuple for OpenCV
    """
    color_map = {
        "Standing": (0, 255, 0),      # Green
        "Sitting": (255, 165, 0),      # Orange
        "Lying Down": (255, 0, 255),   # Magenta
        "Unknown": (128, 128, 128),    # Gray
    }
    return color_map.get(pose_label, (255, 255, 255))  # White as default


def get_action_color(action_label):
    """
    Get a color for displaying the action label.
    
    Args:
        action_label: The action label string
    
    Returns:
        tuple: BGR color tuple for OpenCV
    """
    color_map = {
        "Jumping": (0, 255, 255),      # Yellow
        "Running": (0, 165, 255),      # Orange
        "Walking": (147, 20, 255),     # Pink
        "Fighting": (0, 0, 255),       # Red
        "Idle": (0, 255, 0),           # Green
        "Unknown": (128, 128, 128),    # Gray
    }
    return color_map.get(action_label, (255, 255, 255))  # White as default


class TemporalActionRecognizer:
    """
    Recognizes actions based on temporal analysis of keypoint sequences.
    
    This class maintains a buffer of recent keypoint frames and computes
    temporal features like velocity and acceleration to detect dynamic actions.
    """
    
    def __init__(self, window_size=30, fps=30):
        """
        Initialize the temporal action recognizer.
        
        Args:
            window_size: Number of frames to buffer for temporal analysis
            fps: Frames per second (used for velocity/acceleration calculations)
        """
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps  # Time delta between frames
        
        # Buffer to store keypoint history (deque for efficient operations)
        self.keypoint_buffer = deque(maxlen=window_size)
        
        # Minimum frames needed for temporal analysis
        self.min_frames = 10

        # Action smoothing state
        history_length = max(5, int(self.fps * 0.5))
        self.action_history = deque(maxlen=history_length)
        self.current_action = "Unknown"
        self.dynamic_hold_frames = max(3, int(self.fps * 0.3))
        self.frames_since_dynamic = self.dynamic_hold_frames
        
    def update(self, keypoints):
        """
        Update the buffer with new keypoints and classify the current action.
        
        Args:
            keypoints: numpy array of shape (17, 3) where each row is [y, x, confidence]
        
        Returns:
            str: Detected action label
        """
        if keypoints is None or len(keypoints) != 17:
            return "Unknown"
        
        # Add to buffer
        self.keypoint_buffer.append(keypoints.copy())
        
        # Need enough frames for temporal analysis
        if len(self.keypoint_buffer) < self.min_frames:
            return "Unknown"
        
        # Classify action based on temporal features
        raw_action = self._classify_action()
        return self._stabilize_action(raw_action)
    
    def _get_keypoint_velocity(self, keypoint_idx, num_frames=5):
        """
        Calculate the velocity of a specific keypoint over recent frames.
        
        Args:
            keypoint_idx: Index of the keypoint (0-16)
            num_frames: Number of recent frames to analyze
        
        Returns:
            tuple: (vy, vx, magnitude) - velocity in y, x, and magnitude
        """
        if len(self.keypoint_buffer) < num_frames:
            return 0.0, 0.0, 0.0
        
        # Get recent positions
        recent_frames = list(self.keypoint_buffer)[-num_frames:]
        
        # Calculate average velocity
        velocities_y = []
        velocities_x = []
        
        for i in range(1, len(recent_frames)):
            prev_kp = recent_frames[i-1][keypoint_idx]
            curr_kp = recent_frames[i][keypoint_idx]
            
            # Only calculate if both keypoints are confident
            if prev_kp[2] > CONFIDENCE_THRESHOLD and curr_kp[2] > CONFIDENCE_THRESHOLD:
                vy = (curr_kp[0] - prev_kp[0]) / self.dt
                vx = (curr_kp[1] - prev_kp[1]) / self.dt
                velocities_y.append(vy)
                velocities_x.append(vx)
        
        if not velocities_y:
            return 0.0, 0.0, 0.0
        
        avg_vy = np.mean(velocities_y)
        avg_vx = np.mean(velocities_x)
        magnitude = np.sqrt(avg_vy**2 + avg_vx**2)
        
        return avg_vy, avg_vx, magnitude
    
    def _get_body_center_velocity(self):
        """
        Calculate velocity of the body's center of mass (average of hips and shoulders).
        
        Returns:
            tuple: (vy, vx, magnitude) - velocity of body center
        """
        if len(self.keypoint_buffer) < 5:
            return 0.0, 0.0, 0.0
        
        def get_body_center(keypoints):
            """Calculate body center from shoulders and hips."""
            centers = []
            for idx in [KEYPOINT_LEFT_SHOULDER, KEYPOINT_RIGHT_SHOULDER, 
                       KEYPOINT_LEFT_HIP, KEYPOINT_RIGHT_HIP]:
                kp = keypoints[idx]
                if kp[2] > CONFIDENCE_THRESHOLD:
                    centers.append(kp[:2])
            
            if centers:
                return np.mean(centers, axis=0)
            return None
        
        recent_frames = list(self.keypoint_buffer)[-10:]
        centers = [get_body_center(frame) for frame in recent_frames]
        centers = [c for c in centers if c is not None]
        
        if len(centers) < 2:
            return 0.0, 0.0, 0.0
        
        # Calculate velocity
        velocities_y = []
        velocities_x = []
        
        for i in range(1, len(centers)):
            vy = (centers[i][0] - centers[i-1][0]) / self.dt
            vx = (centers[i][1] - centers[i-1][1]) / self.dt
            velocities_y.append(vy)
            velocities_x.append(vx)
        
        avg_vy = np.mean(velocities_y)
        avg_vx = np.mean(velocities_x)
        magnitude = np.sqrt(avg_vy**2 + avg_vx**2)
        
        return avg_vy, avg_vx, magnitude
    
    def _are_feet_off_ground(self):
        """
        Detect if both ankles are moving upward (jump initiation cue).
        
        Returns:
            bool: True if both ankles exhibit upward velocity
        """
        if len(self.keypoint_buffer) < 3:
            return False

        left_vy, _, _ = self._get_keypoint_velocity(KEYPOINT_LEFT_ANKLE, num_frames=4)
        right_vy, _, _ = self._get_keypoint_velocity(KEYPOINT_RIGHT_ANKLE, num_frames=4)

        upward_threshold = -0.35  # Negative vy means upward motion

        return left_vy < upward_threshold and right_vy < upward_threshold
    
    def _detect_alternating_leg_motion(self):
        """
        Detect alternating leg motion (for walking/running detection).
        
        Returns:
            bool: True if alternating leg motion detected
        """
        if len(self.keypoint_buffer) < 12:
            return False
        
        recent_frames = list(self.keypoint_buffer)[-15:]

        left_knee_positions = []
        right_knee_positions = []

        for frame in recent_frames:
            left_knee = frame[KEYPOINT_LEFT_KNEE]
            right_knee = frame[KEYPOINT_RIGHT_KNEE]

            if left_knee[2] > CONFIDENCE_THRESHOLD and right_knee[2] > CONFIDENCE_THRESHOLD:
                left_knee_positions.append(left_knee[0])
                right_knee_positions.append(right_knee[0])

        if len(left_knee_positions) < 8:
            return False

        left = np.array(left_knee_positions)
        right = np.array(right_knee_positions)

        if np.std(left) < 0.01 and np.std(right) < 0.01:
            return False

        diff = left - right
        if np.std(diff) < 0.015:
            return False

        signs = np.sign(diff)
        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]

        sign_changes = np.sum(signs[1:] != signs[:-1])
        has_alternation = sign_changes >= 2

        if len(left) > 1:
            correlation = np.corrcoef(left, right)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        return has_alternation and correlation < 0.2
    
    def _detect_arm_strike(self):
        """
        Detect rapid arm motion indicative of a forward strike.
        
        Returns:
            bool: True if either arm is moving rapidly enough to indicate a strike.
        """
        _, _, left_wrist_mag = self._get_keypoint_velocity(KEYPOINT_LEFT_WRIST, num_frames=5)
        _, _, right_wrist_mag = self._get_keypoint_velocity(KEYPOINT_RIGHT_WRIST, num_frames=5)

        strike_threshold = 1.6  # Slightly relaxed to register single-arm strikes

        return left_wrist_mag > strike_threshold or right_wrist_mag > strike_threshold
    
    def _classify_action(self):
        """
        Classify the current action based on temporal features.
        
        Returns:
            str: Action label
        """
        # Get current static pose
        current_keypoints = self.keypoint_buffer[-1]
        static_pose = estimate_pose(current_keypoints)
        
        # Calculate body center velocity
        body_vy, body_vx, body_velocity = self._get_body_center_velocity()
        
        # Get feet velocity
        left_ankle_vy, left_ankle_vx, left_ankle_vel = self._get_keypoint_velocity(KEYPOINT_LEFT_ANKLE)
        right_ankle_vy, right_ankle_vx, right_ankle_vel = self._get_keypoint_velocity(KEYPOINT_RIGHT_ANKLE)
        avg_foot_velocity = (left_ankle_vel + right_ankle_vel) / 2

        alternating_motion = self._detect_alternating_leg_motion()
        
        # Detect specific actions (order matters - check most distinctive first)
        
        # 1. JUMPING: Feet off ground
        if self._are_feet_off_ground():
            if body_vy < -0.2:  # Negative vy = upward motion
                return "Jumping"
        # 2. FIGHTING: Any rapid arm strike motion
        # maintain jumping priority over fighting
        elif self._detect_arm_strike():
            return "Fighting"
        
        # 3. RUNNING: High body velocity or fast feet, plus alternating legs
        if static_pose in ("Standing", "Unknown"):
            if abs(body_vx) > 0.6:
                #if alternating_motion or avg_foot_velocity > 0.8:
                return "Running"

        # 4. WALKING: Moderate body velocity with alternating legs
        if static_pose in ("Standing", "Unknown"):
            if abs(body_vx) > 0.2:
                #if alternating_motion or avg_foot_velocity > 0.5:
                return "Walking"

        # Allow alternating motion alone to keep walking active when velocity dips slightly
        if alternating_motion and avg_foot_velocity > 0.4 and static_pose == "Standing":
            return "Walking"

        # 5. IDLE: Minimal motion
        if body_velocity < 0.25 and avg_foot_velocity < 0.4:
            return "Idle"
        
        # Default: return static pose or unknown
        return static_pose if static_pose != "Unknown" else "Unknown"

    def _stabilize_action(self, raw_action):
        """Stabilize the instantaneous action prediction across frames."""
        dynamic_actions = {"Running", "Walking", "Jumping", "Fighting"}
        static_actions = {"Standing", "Sitting", "Lying Down", "Idle"}

        self.action_history.append(raw_action)

        history_dynamic = [a for a in self.action_history if a in dynamic_actions]
        if history_dynamic:
            candidate = max(set(history_dynamic), key=history_dynamic.count)
            min_votes = max(2, int(len(self.action_history) * 0.4))
            if history_dynamic.count(candidate) >= min_votes:
                self.current_action = candidate
                self.frames_since_dynamic = 0
                return candidate

        if raw_action in dynamic_actions:
            self.current_action = raw_action
            self.frames_since_dynamic = 0
            return raw_action

        if self.current_action in dynamic_actions:
            self.frames_since_dynamic += 1
            if self.frames_since_dynamic <= self.dynamic_hold_frames:
                return self.current_action
            self.current_action = "Unknown"

        history_static = [a for a in self.action_history if a in static_actions]
        if history_static:
            candidate = max(set(history_static), key=history_static.count)
            self.current_action = candidate
            return candidate

        self.current_action = raw_action
        return raw_action
