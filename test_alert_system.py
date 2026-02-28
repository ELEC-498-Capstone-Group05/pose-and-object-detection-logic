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


def test_monitoring_failure_triggers_on_visual_obstruction():
    app = Flask(__name__)
    alert = AlertSystem(app)

    cfg = alert.alerts_config["monitoring_failure"]
    cfg["obstruction_persist_frames"] = 1
    cfg["dark_mean_threshold"] = 255.0
    cfg["blur_var_threshold"] = 1000000.0

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with patch.object(alert.socketio, "emit") as emit_mock:
        alert.process_frame(
            yolo_results=([], [], [], None),
            movenet_results=_build_keypoints(),
            action_label="Idle",
            input_size=(640, 640),
            frame=frame,
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

    with patch.object(alert.socketio, "emit") as emit_mock:
        alert.process_frame(
            yolo_results=([], [], [], None),
            movenet_results=None,
            action_label="Fighting",
            input_size=(640, 640),
        )

    payload = emit_mock.call_args.args[1]
    assert payload["type"] == "fighting"
    assert payload["severity"] == "medium"
