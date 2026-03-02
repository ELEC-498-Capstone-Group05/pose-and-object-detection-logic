import time
from unittest.mock import patch

import numpy as np
from flask import Flask

from alert_system import AlertSystem


def _build_keypoints():
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 2] = 1.0
    keypoints[0] = [0.4, 0.5, 1.0]  # nose
    return keypoints


def _collect_alert_types(mock_emit):
    types = []
    for call in mock_emit.call_args_list:
        args = call.args
        if len(args) >= 2 and args[0] == "alert":
            payload = args[1]
            types.append(payload.get("type"))
    return types


def _count_alert_type(mock_emit, alert_type):
    return sum(1 for t in _collect_alert_types(mock_emit) if t == alert_type)


def test_fall_no_recovery_triggers_after_timeout():
    app = Flask(__name__)
    alert = AlertSystem(app)

    alert.alerts_config["fall"]["enabled"] = False
    alert.alerts_config["fall_no_recovery"]["duration_s"] = 0.0
    with patch.object(alert.socketio, "emit") as emit_mock:
        alert.process_frame(
            yolo_results=([], [], [], None),
            movenet_results=None,
            action_label="Fall Detected",
            input_size=(640, 640),
        )

    assert "fall_no_recovery" in _collect_alert_types(emit_mock)


def test_fall_alert_triggers_only_once_per_episode():
    app = Flask(__name__)
    alert = AlertSystem(app)

    alert.alerts_config["fall_no_recovery"]["enabled"] = False

    clock = [100.0]

    def fake_time():
        return clock[0]

    with patch("alert_system.time.time", side_effect=fake_time):
        with patch.object(alert.socketio, "emit") as emit_mock:
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Fall Detected",
                input_size=(640, 640),
            )
            clock[0] += 4.0
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Fall Detected",
                input_size=(640, 640),
            )
            clock[0] += 4.0
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Fall Detected",
                input_size=(640, 640),
            )

    assert _count_alert_type(emit_mock, "fall") == 1


def test_fall_alert_requires_stable_recovery_before_new_episode():
    app = Flask(__name__)
    alert = AlertSystem(app)

    alert.alerts_config["fall_no_recovery"]["enabled"] = False
    alert.alerts_config["fall"]["recovery_stable_s"] = 1.0

    clock = [200.0]

    def fake_time():
        return clock[0]

    with patch("alert_system.time.time", side_effect=fake_time):
        with patch.object(alert.socketio, "emit") as emit_mock:
            # Episode 1 starts.
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Fall Detected",
                input_size=(640, 640),
            )

            # Brief upright jitter should not end the episode.
            clock[0] += 0.4
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Standing",
                input_size=(640, 640),
            )
            clock[0] += 0.2
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Fall Detected",
                input_size=(640, 640),
            )

            # Stable recovery window completes.
            clock[0] += 0.1
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Standing",
                input_size=(640, 640),
            )
            clock[0] += 1.1
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Standing",
                input_size=(640, 640),
            )

            # Episode 2 should now be allowed.
            clock[0] += 0.1
            alert.process_frame(
                yolo_results=([], [], [], None),
                movenet_results=None,
                action_label="Fall Detected",
                input_size=(640, 640),
            )

    assert _count_alert_type(emit_mock, "fall") == 2


def test_fall_no_recovery_triggers_once_per_episode():
    app = Flask(__name__)
    alert = AlertSystem(app)

    alert.alerts_config["fall"]["enabled"] = False
    alert.alerts_config["fall"]["recovery_stable_s"] = 1.0
    alert.alerts_config["fall_no_recovery"]["duration_s"] = 2.0
    alert.alerts_config["fall_no_recovery"]["cooldown_s"] = 0.0

    clock = [300.0]

    def fake_time():
        return clock[0]

    with patch("alert_system.time.time", side_effect=fake_time):
        with patch.object(alert.socketio, "emit") as emit_mock:
            # One sustained fall episode should emit no-recovery only once.
            alert.process_frame(([], [], [], None), None, "Fall Detected", (640, 640))
            clock[0] += 2.1
            alert.process_frame(([], [], [], None), None, "Fall Detected", (640, 640))
            clock[0] += 5.0
            alert.process_frame(([], [], [], None), None, "Fall Detected", (640, 640))

    assert _count_alert_type(emit_mock, "fall_no_recovery") == 1


def test_choking_cough_distress_triggers_on_burst():
    app = Flask(__name__)
    alert = AlertSystem(app)

    alert.alerts_config["choking_cough_distress"]["min_count"] = 2
    alert.alerts_config["choking_cough_distress"]["window_s"] = 10.0
    with patch.object(alert.socketio, "emit") as emit_mock:
        t0 = time.time()
        cough_detection = {42: {"score": 0.9, "name": "Cough"}}
        alert.process_audio(cough_detection, audio_db=-40.0, timestamp=t0)
        alert.process_audio(cough_detection, audio_db=-40.0, timestamp=t0 + 1.0)

    assert "choking_cough_distress" in _collect_alert_types(emit_mock)


def test_unusual_silence_triggers_when_person_recent_and_no_motion():
    app = Flask(__name__)
    alert = AlertSystem(app)

    cfg = alert.alerts_config["unusual_silence"]
    cfg["silence_duration_s"] = 0.0
    cfg["motion_grace_s"] = 0.0
    cfg["person_required_within_s"] = 60.0

    now = time.time()
    alert.temporal_state["activity"]["last_person_seen_ts"] = now
    alert.temporal_state["activity"]["last_motion_ts"] = now - 120.0

    with patch.object(alert.socketio, "emit") as emit_mock:
        alert.process_audio({}, audio_db=-80.0, timestamp=now + 1.0)

    assert "unusual_silence" in _collect_alert_types(emit_mock)


def test_monitoring_failure_triggers_on_no_objects_duration():
    app = Flask(__name__)
    alert = AlertSystem(app)

    cfg = alert.alerts_config["monitoring_failure"]
    cfg["no_objects_duration_s"] = 0.0

    with patch.object(alert.socketio, "emit") as emit_mock:
        alert.process_frame(
            yolo_results=([], [], [], None),
            movenet_results=_build_keypoints(),
            action_label="Idle",
            input_size=(640, 640),
            monitoring_context={
                "camera_read_failures": 0,
                "yolo_result_age_frames": 0,
                "movenet_result_age_frames": 0,
            },
        )

    assert "monitoring_failure" in _collect_alert_types(emit_mock)


def test_alert_payload_includes_severity():
    app = Flask(__name__)
    alert = AlertSystem(app)

    # Fighting alert now requires at least 2 detected people.
    yolo_results = (
        np.array([[0.2, 0.3, 0.2, 0.2], [0.7, 0.3, 0.2, 0.2]], dtype=np.float32),
        np.array([0, 0], dtype=np.int32),
        np.array([0.9, 0.85], dtype=np.float32),
        None,
    )

    with patch.object(alert.socketio, "emit") as emit_mock:
        alert.process_frame(
            yolo_results=yolo_results,
            movenet_results=None,
            action_label="Fighting",
            input_size=(640, 640),
        )

    payload = emit_mock.call_args.args[1]
    assert payload["type"] == "fighting"
    assert payload["severity"] == "medium"


def test_fighting_alert_not_triggered_with_only_one_person():
    app = Flask(__name__)
    alert = AlertSystem(app)

    yolo_results = (
        np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
        np.array([0], dtype=np.int32),
        np.array([0.95], dtype=np.float32),
        None,
    )

    with patch.object(alert.socketio, "emit") as emit_mock:
        alert.process_frame(
            yolo_results=yolo_results,
            movenet_results=None,
            action_label="Fighting",
            input_size=(640, 640),
        )

    assert "fighting" not in _collect_alert_types(emit_mock)
