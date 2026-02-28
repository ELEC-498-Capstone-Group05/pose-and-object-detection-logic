import numpy as np

from pose_estimator import (
    KEYPOINT_LEFT_HIP,
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
