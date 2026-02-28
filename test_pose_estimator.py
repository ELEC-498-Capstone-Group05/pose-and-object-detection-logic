import time
import numpy as np

from pose_estimator import (
    KEYPOINT_NOSE,
    KEYPOINT_LEFT_ANKLE,
    KEYPOINT_LEFT_HIP,
    KEYPOINT_LEFT_WRIST,
    KEYPOINT_RIGHT_ANKLE,
    KEYPOINT_LEFT_SHOULDER,
    KEYPOINT_RIGHT_HIP,
    KEYPOINT_RIGHT_SHOULDER,
    TemporalActionRecognizer,
    estimate_pose,
    get_pose_color,
)


def _base_keypoints(conf=1.0):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 2] = conf
    return keypoints


def _set_torso(keypoints, center_y, center_x, confidence=1.0):
    # Place a compact torso around the center.
    keypoints[KEYPOINT_LEFT_SHOULDER] = [center_y - 0.08, center_x - 0.05, confidence]
    keypoints[KEYPOINT_RIGHT_SHOULDER] = [center_y - 0.08, center_x + 0.05, confidence]
    keypoints[KEYPOINT_LEFT_HIP] = [center_y + 0.08, center_x - 0.05, confidence]
    keypoints[KEYPOINT_RIGHT_HIP] = [center_y + 0.08, center_x + 0.05, confidence]
    return keypoints


def _set_sparse_torso(keypoints):
    # Keep only one torso keypoint confident so center quorum is not met.
    keypoints[KEYPOINT_LEFT_SHOULDER] = [0.42, 0.45, 1.0]
    keypoints[KEYPOINT_RIGHT_SHOULDER] = [0.42, 0.55, 0.1]
    keypoints[KEYPOINT_LEFT_HIP] = [0.58, 0.45, 0.1]
    keypoints[KEYPOINT_RIGHT_HIP] = [0.58, 0.55, 0.1]
    return keypoints


def _set_standing_limbs(keypoints):
    keypoints[KEYPOINT_LEFT_WRIST] = [0.45, 0.40, 1.0]
    keypoints[KEYPOINT_LEFT_ANKLE] = [0.78, 0.44, 1.0]
    keypoints[KEYPOINT_RIGHT_ANKLE] = [0.78, 0.56, 1.0]
    return keypoints


def _build_standing_frame(center_y=0.5, center_x=0.5):
    frame = _set_torso(_base_keypoints(), center_y=center_y, center_x=center_x)
    return _set_standing_limbs(frame)


def test_estimate_pose_upside_down():
    keypoints = _base_keypoints()
    keypoints[KEYPOINT_LEFT_SHOULDER] = [0.60, 0.45, 1.0]
    keypoints[KEYPOINT_RIGHT_SHOULDER] = [0.60, 0.55, 1.0]
    keypoints[KEYPOINT_LEFT_HIP] = [0.45, 0.45, 1.0]
    keypoints[KEYPOINT_RIGHT_HIP] = [0.45, 0.55, 1.0]

    assert estimate_pose(keypoints) == "Upside Down"


def test_get_pose_color_has_upside_down_class():
    assert get_pose_color("Upside Down") == (255, 255, 0)


def test_body_center_velocity_uses_real_frame_gap_on_dropouts():
    recognizer = TemporalActionRecognizer(window_size=30, fps=10)

    frame_0 = _set_torso(_base_keypoints(), center_y=0.5, center_x=0.5)
    frame_1 = _set_sparse_torso(_base_keypoints())
    frame_2 = _set_torso(_base_keypoints(), center_y=0.5, center_x=0.6)
    frame_3 = _set_sparse_torso(_base_keypoints())
    frame_4 = _set_sparse_torso(_base_keypoints())

    for frame in [frame_0, frame_1, frame_2, frame_3, frame_4]:
        recognizer.keypoint_buffer.append(frame)

    vy, vx, speed = recognizer._get_body_center_velocity()

    # center_x moved +0.1 over 2 frame intervals at fps=10 => 0.1 / (2 * 0.1) = 0.5
    assert abs(vy) < 1e-6
    assert np.isclose(vx, 0.5, atol=1e-6)
    assert np.isclose(speed, 0.5, atol=1e-6)


def test_stabilize_action_tie_break_is_deterministic():
    outputs = []
    for _ in range(20):
        recognizer = TemporalActionRecognizer(window_size=30, fps=30)
        recognizer.action_history.extend(["Walking", "Running", "Walking", "Running"])
        outputs.append(recognizer._stabilize_action("Unknown"))

    assert all(label == "Running" for label in outputs)


def test_jumping_onset_tracks_latency_with_frame_timestamps():
    recognizer = TemporalActionRecognizer(window_size=30, fps=30)

    # Warm-up frames so temporal logic is active.
    for seq in range(9):
        frame = _build_standing_frame(center_y=0.50)
        recognizer.update(
            frame,
            static_pose="Standing",
            frame_seq=seq,
            capture_ts=1000.0,
        )

    # Upward ankle and body center movement should trigger jumping.
    # Keep motion strongly upward for jumping, but below kick speed thresholds.
    jump_centers = [0.49, 0.46, 0.43, 0.40]
    jump_ankle_ys = [0.74, 0.71, 0.68, 0.65]
    action = "Unknown"

    for offset, (center_y, ankle_y) in enumerate(zip(jump_centers, jump_ankle_ys), start=9):
        frame = _build_standing_frame(center_y=center_y)
        frame[KEYPOINT_LEFT_ANKLE] = [ankle_y, 0.44, 1.0]
        frame[KEYPOINT_RIGHT_ANKLE] = [ankle_y, 0.56, 1.0]
        action = recognizer.update(
            frame,
            static_pose="Standing",
            frame_seq=offset,
            capture_ts=time.perf_counter() - 0.05,
        )
        if action == "Jumping":
            break

    assert action == "Jumping"
    assert recognizer.last_onset_action == "Jumping"
    assert recognizer.last_onset_frame_seq is not None
    assert recognizer.last_onset_latency_ms is not None
    assert recognizer.last_onset_latency_ms >= 0.0


def test_punching_onset_handles_missing_capture_timestamp():
    recognizer = TemporalActionRecognizer(window_size=30, fps=30)

    wrist_positions = [0.40] * 9 + [0.44, 0.50, 0.58]
    action = "Unknown"

    for seq, wrist_x in enumerate(wrist_positions):
        frame = _build_standing_frame(center_y=0.50)
        frame[KEYPOINT_LEFT_WRIST] = [0.45, wrist_x, 1.0]
        action = recognizer.update(
            frame,
            static_pose="Standing",
            frame_seq=seq,
            capture_ts=None,
        )

    assert action in {"Punching", "Fighting"}
    assert recognizer.last_onset_action == "Punching"
    assert recognizer.last_onset_latency_ms is None


def test_head_impact_action_detected_on_abrupt_downward_motion():
    recognizer = TemporalActionRecognizer(window_size=30, fps=30)

    # Warm-up with stable standing frames.
    for seq in range(9):
        frame = _build_standing_frame(center_y=0.46)
        frame[KEYPOINT_NOSE] = [0.34, 0.50, 1.0]
        recognizer.update(frame, static_pose="Standing", frame_seq=seq, capture_ts=None)

    action = "Unknown"
    # Rapid downward body and nose movement to trigger head-impact heuristic.
    centers = [0.52, 0.60, 0.68]
    nose_ys = [0.48, 0.60, 0.73]
    for seq, (center_y, nose_y) in enumerate(zip(centers, nose_ys), start=9):
        frame = _build_standing_frame(center_y=center_y)
        frame[KEYPOINT_NOSE] = [nose_y, 0.50, 1.0]
        action = recognizer.update(frame, static_pose="Standing", frame_seq=seq, capture_ts=None)
        if action == "Head Impact Suspected":
            break

    assert action == "Head Impact Suspected"
