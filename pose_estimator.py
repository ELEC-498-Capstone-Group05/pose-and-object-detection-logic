"""
Pose Estimator for MoveNet Keypoints

This module analyzes the 17 keypoints from MoveNet to estimate basic human poses
and complex actions over time.

Features:
- Static pose estimation (standing, sitting, lying down, bending)
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
        str: Estimated pose label - "Standing", "Sitting", "Lying Down", "Bending" or "Unknown"
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
     # Check for upside down: hips above shoulders
    if hip_y < shoulder_y - 0.05:
        return "Upside Down"
    
    # Calculate torso length (vertical distance from shoulders to hips)
    torso_length = abs(hip_y - shoulder_y)
    # Calculate hip to knee distance if knees are available
    hip_to_knee = None
    if knee_y is not None:
        hip_to_knee = knee_y - hip_y  # Positive means knees are below hips

    # Analyze pose based on body geometry
    # Remember: lower y values = higher on screen (top of image)
    # Check for lying down: torso is nearly horizontal (small vertical distance)
    if torso_length < 0.15:
        if hip_to_knee is not None:
            if abs(hip_to_knee) < 0.1:
                return "Lying Down"
            elif hip_to_knee > 0.15:
                return "Bending"
            else:
                return "Lying Down"
        else:
            return "Lying Down"
    # Torso is more vertical - check for standing vs sitting
    else:
        # Analyze knee position relative to hips
        if knee_y is not None:
            # Standing: knees significantly below hips
            if hip_to_knee > 0.15:
                return "Standing"
            # Sitting: knees are roughly at same level or above hips
            else:
                return "Sitting"
        else:
            return "Standing"

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
        "Bending": (0, 128, 255),
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
        
    def update(self, keypoints, static_pose=None):
        """
        Update the buffer with new keypoints and classify the current action.
        
        Args:
            keypoints: numpy array of shape (17, 3) where each row is [y, x, confidence]
            static_pose: Optional pre-calculated static pose label. If None, it will be calculated.
        
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
        raw_action = self._classify_action(static_pose)
        return self._stabilize_action(raw_action)
    
    def _get_keypoint_velocity(self, keypoint_idx, num_frames=5):
        """
        Calculate the velocity of a specific keypoint over recent frames.
        
        Args:
            keypoint_idx: Index of the keypoint (0-16)
            num_frames: Number of recent frames to analyze
        
        Returns:
            tuple: (vy, vx, speed) - velocity in y, x, and scalar speed
        """
        if len(self.keypoint_buffer) < num_frames:
            return 0.0, 0.0, 0.0
        
        # Get recent positions
        recent_frames = list(self.keypoint_buffer)[-num_frames:]
        
        # Calculate velocities
        velocities_y = []
        velocities_x = []
        element_speeds = []
        
        for i in range(1, len(recent_frames)):
            prev_kp = recent_frames[i-1][keypoint_idx]
            curr_kp = recent_frames[i][keypoint_idx]
            
            # Only calculate if both keypoints are confident
            if prev_kp[2] > CONFIDENCE_THRESHOLD and curr_kp[2] > CONFIDENCE_THRESHOLD:
                vy = (curr_kp[0] - prev_kp[0]) / self.dt
                vx = (curr_kp[1] - prev_kp[1]) / self.dt
                velocities_y.append(vy)
                velocities_x.append(vx)
                element_speeds.append(np.sqrt(vy**2 + vx**2))
        
        if not velocities_y:
            return 0.0, 0.0, 0.0
        
        # Return average directional velocity, use MEAN speed to avoid noise spikes
        avg_vy = np.mean(velocities_y)
        avg_vx = np.mean(velocities_x)
        
        # Use mean speed for more robust detection (original was vector magnitude)
        speed = np.mean(element_speeds) if element_speeds else 0.0
        
        return avg_vy, avg_vx, speed
    
    def _get_keypoint_displacement(self, keypoint_idx, num_frames=5):
        """
        Calculate the euclidean distance between the first and last frame in the window.
        Use this to filter out jitter where velocity is high but the point hasn't moved.
        """
        if len(self.keypoint_buffer) < num_frames:
            return 0.0
            
        recent_frames = list(self.keypoint_buffer)[-num_frames:]
        start_kp = recent_frames[0][keypoint_idx]
        end_kp = recent_frames[-1][keypoint_idx]
        
        # Check confidence
        if start_kp[2] < CONFIDENCE_THRESHOLD or end_kp[2] < CONFIDENCE_THRESHOLD:
            return 0.0
            
        dy = end_kp[0] - start_kp[0]
        dx = end_kp[1] - start_kp[1]
        
        return np.sqrt(dy**2 + dx**2)
    
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
    
    def _feet_are_moving_upward(self):
        """
        Detect if both ankles are moving upward (jump initiation cue).
        
        Returns:
            bool: True if both ankles exhibit upward velocity
        """
        if len(self.keypoint_buffer) < 3:
            return False

        left_vy, _, _ = self._get_keypoint_velocity(KEYPOINT_LEFT_ANKLE, num_frames=4)
        right_vy, _, _ = self._get_keypoint_velocity(KEYPOINT_RIGHT_ANKLE, num_frames=4)

        upward_threshold = -0.1  # Negative vy means upward motion

        return left_vy < upward_threshold and right_vy < upward_threshold
    
    def _detect_arm_strike(self):
        """
        Detect rapid arm motion indicative of a forward strike.
        
        Returns:
            bool: True if either arm is moving rapidly enough to indicate a strike.
        """
        # Reduced frames to 3 to catch the peak of the punch
        _, _, left_wrist_speed = self._get_keypoint_velocity(KEYPOINT_LEFT_WRIST, num_frames=3)
        _, _, right_wrist_speed = self._get_keypoint_velocity(KEYPOINT_RIGHT_WRIST, num_frames=3)
        
        # Check total displacement to filter out jitter/noise
        left_disp = self._get_keypoint_displacement(KEYPOINT_LEFT_WRIST, num_frames=5)
        right_disp = self._get_keypoint_displacement(KEYPOINT_RIGHT_WRIST, num_frames=5)

        # Higher threshold to reduce false positives
        strike_threshold = 2.0
        displacement_threshold = 0.1

        left_strike = left_wrist_speed > strike_threshold and left_disp > displacement_threshold
        right_strike = right_wrist_speed > strike_threshold and right_disp > displacement_threshold

        return left_strike or right_strike
    
    def _detect_leg_kick(self):
        """
        Detect rapid leg motion indicative of a kicking action.
        
        Returns:
            bool: True if either leg is moving rapidly enough to indicate a kick.
        """
        _, _, left_ankle_speed = self._get_keypoint_velocity(KEYPOINT_LEFT_ANKLE, num_frames=3)
        _, _, right_ankle_speed = self._get_keypoint_velocity(KEYPOINT_RIGHT_ANKLE, num_frames=3)
        
        # Check total displacement to filter out jitter/noise
        left_disp = self._get_keypoint_displacement(KEYPOINT_LEFT_ANKLE, num_frames=5)
        right_disp = self._get_keypoint_displacement(KEYPOINT_RIGHT_ANKLE, num_frames=5)

        # Higher threshold to reduce false positives
        kick_threshold = 2.0
        displacement_threshold = 0.1

        left_kick = left_ankle_speed > kick_threshold and left_disp > displacement_threshold
        right_kick = right_ankle_speed > kick_threshold and right_disp > displacement_threshold

        return left_kick or right_kick

    def _classify_action(self, static_pose=None):
        """
        Classify the current action based on temporal features.
        
        Args:
            static_pose: Optional pre-calculated static pose label.
        
        Returns:
            str: Action label
        """
        if static_pose is None:
            # Get current static pose from buffer if not provided
            current_keypoints = self.keypoint_buffer[-1]
            static_pose = estimate_pose(current_keypoints)
        if static_pose == "Unknown":
            return "Unknown"
        
        # Calculate body center velocity
        body_vy, body_vx, body_velocity = self._get_body_center_velocity()
        
        # Detect specific actions (order matters - check most distinctive first)
        # 1. JUMPING: Feet are moving upward rapidly and body is ascending
        if  self._feet_are_moving_upward() and body_vy < -0.1:
            return "Jumping"
            
        # 2. FIGHTING: Any rapid arm strike motion
        # maintain jumping priority over fighting
        if static_pose == "Standing" and (self._detect_arm_strike() or self._detect_leg_kick()):
            return "Fighting"
    
        # 3. RUNNING / WALKING: moving horizontally
        if static_pose == "Standing":
            if abs(body_vx) > 0.3:
                return "Running"
            elif abs(body_vx) > 0.1:
                return "Walking"
            
        # 4. IDLE: Minimal motion
        if body_velocity < 0.25:
            return "Idle"
        
        # Default: return static pose or unknown
        return static_pose if static_pose != "Unknown" else "Unknown"

    def _stabilize_action(self, raw_action):
        """Stabilize the instantaneous action prediction across frames."""
        dynamic_actions = {"Running", "Walking", "Jumping", "Fighting"}
        static_actions = {"Standing", "Sitting", "Lying Down", "Idle"}
        
        # Special logic for Fighting: it's an impulse action, so we hold it longer
        # If we just detected Fighting, force it to be the current action
        if raw_action == "Fighting":
            self.current_action = "Fighting"
            self.frames_since_dynamic = -15  # Negative value gives it extra "stickiness" (approx 0.5s extra)
            self.action_history.append(raw_action)
            return "Fighting"

        self.action_history.append(raw_action)

        # Logic to hold dynamic actions
        if self.current_action == "Fighting":
             self.frames_since_dynamic += 1
             # Use a longer hold for fighting specifically (approx 30 frames / 1 sec total)
             fighting_hold_limit = max(30, int(self.fps * 1.0)) 
             if self.frames_since_dynamic <= fighting_hold_limit:
                 return "Fighting"

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
