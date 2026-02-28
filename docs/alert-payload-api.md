# Mobile App Integration: Alert Payload (Socket.IO)

This document defines the real-time alert payload emitted by `alert_system.py` for the mobile app.

## Summary

- Transport: Socket.IO
- Event name: `alert`
- Server endpoint: same host/port as Flask (`http://<PI_IP>:5000`)
- Timestamp unit: Unix epoch **milliseconds** (`number`)

## Wire Payload

Each emitted alert follows this base shape:

```json
{
  "type": "fall_no_recovery",
  "message": "Fall detected with no recovery!",
  "severity": "high",
  "timestamp": 1761654325123,
  "metadata": {
    "reason": "no_recovery_timeout",
    "elapsed_s": 8.12
  }
}
```

### Fields

- `type` (`string`, required): internal alert key (stable identifier for app logic)
- `message` (`string`, required): human-readable alert text
- `severity` (`"low" | "medium" | "high"`, required): priority level for notification UX
- `timestamp` (`number`, required): Unix epoch time in milliseconds
- `metadata` (`object`, optional): context for the specific alert type

## Alert Type Values

`type` currently uses these keys from `alert_system.py`:

- `fighting`
- `fall`
- `fall_no_recovery`
- `head_impact_suspected`
- `knife`
- `weapon`
- `knife_in_hand`
- `jumping_on_couch`
- `climbing_hazard`
- `screaming`
- `crying`
- `shatter`
- `choking_cough_distress`
- `unusual_silence`
- `safezone_exit`
- `monitoring_failure`

## Metadata By Alert

`metadata` is only included for alerts that provide extra diagnostic context.

### `fall_no_recovery`

```json
{
  "reason": "no_recovery_timeout",
  "elapsed_s": 8.34
}
```

### `climbing_hazard`

```json
{
  "reason": "sustained_upward_motion",
  "upward_delta": 0.142,
  "samples": 9,
  "furniture_overlap": true
}
```

### `choking_cough_distress`

```json
{
  "reason": "cough_burst_pattern",
  "count": 4,
  "window_s": 4.0,
  "avg_confidence": 0.61
}
```

### `unusual_silence`

```json
{
  "reason": "fixed_timeout_silence",
  "silence_duration_s": 21.9,
  "audio_db": -59.3
}
```

### `monitoring_failure`

```json
{
  "reason": "stale_inference,visual_obstruction",
  "yolo_result_age_frames": 53,
  "movenet_result_age_frames": 50,
  "camera_read_failures": 0,
  "visual": {
    "mean_luma": 11.2,
    "lap_var": 7.9
  }
}
```

Possible `reason` fragments include:

- `stale_inference`
- `visual_obstruction`
- `camera_read_failures`

For simple alerts (for example `knife`, `safezone_exit`, `fighting`), `metadata` may be absent.

## TypeScript Contracts

```ts
export type AlertType =
  | "fighting"
  | "fall"
  | "fall_no_recovery"
  | "head_impact_suspected"
  | "knife"
  | "weapon"
  | "knife_in_hand"
  | "jumping_on_couch"
  | "climbing_hazard"
  | "screaming"
  | "crying"
  | "shatter"
  | "choking_cough_distress"
  | "unusual_silence"
  | "safezone_exit"
  | "monitoring_failure";

export type AlertPayload = {
  type: AlertType | string;
  message: string;
  severity: "low" | "medium" | "high" | string;
  timestamp: number; // unix ms
  metadata?: Record<string, unknown>;
};
```

## Socket.IO Client Example (React Native)

```ts
import { io, Socket } from "socket.io-client";

const BASE_URL = "http://<PI_IP>:5000";

let socket: Socket | null = null;

export function connectAlerts(onAlert: (a: AlertPayload) => void) {
  socket = io(BASE_URL, {
    transports: ["websocket", "polling"],
  });

  socket.on("connect", () => {
    console.log("Alert socket connected", socket?.id);
  });

  socket.on("alert", (payload: AlertPayload) => {
    if (!payload || typeof payload.type !== "string") return;
    if (typeof payload.message !== "string") return;
    if (typeof payload.severity !== "string") return;
    if (typeof payload.timestamp !== "number") return;
    onAlert(payload);
  });
}

export function disconnectAlerts() {
  socket?.disconnect();
  socket = null;
}
```

## Optional: Poll Alert Health/State

The HTTP `GET /stats` response includes an `alerts` object for per-alert counters and cooldown state.

Useful fields per alert key:

- `enabled` (`boolean`)
- `last_trigger_unix_ms` (`number`)
- `trigger_count` (`number`)
- `suppressed_count` (`number`)
- `cooldown_remaining_ms` (`number`)
- `last_message` (`string | null`)
- `last_reason` (`string | null`)

This is useful for diagnostics dashboards. Real-time user notifications should still use Socket.IO `alert` events.
