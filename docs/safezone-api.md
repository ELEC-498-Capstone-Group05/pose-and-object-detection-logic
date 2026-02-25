# Safezone Configuration API

This document specifies how the mobile app configures the server safezone for "person left safezone" alerts.

## Summary

- Transport: HTTP JSON
- Base server: same host/port as existing Flask API (example: `http://<pi-ip>:5000`)
- Endpoints:
  - `GET /safezone` — read current safezone config
  - `POST /safezone` — create/update safezone config
- Alert channel: existing Socket.IO `alert` event

## Coordinate System

Safezone coordinates use normalized image space:

- `x` is horizontal position from left to right (`0.0` to `1.0`)
- `y` is vertical position from top to bottom (`0.0` to `1.0`)
- `width` and `height` are normalized dimensions (`0.0` to `1.0`)

Rectangle bounds must stay inside frame:

- `x + width <= 1.0`
- `y + height <= 1.0`

## Request: POST /safezone

### JSON payload

```json
{
  "enabled": true,
  "zone": {
    "x": 0.15,
    "y": 0.10,
    "width": 0.55,
    "height": 0.75
  }
}
```

### Field definitions

- `enabled` (required, boolean): enables/disables safezone checking
- `zone` (required, object): rectangle definition
  - `x` (required, number, `[0,1]`)
  - `y` (required, number, `[0,1]`)
  - `width` (required, number, `[0,1]`)
  - `height` (required, number, `[0,1]`)

## Responses

### Success (200)

```json
{
  "ok": true,
  "safezone": {
    "enabled": true,
    "zone": {
      "x": 0.15,
      "y": 0.1,
      "width": 0.55,
      "height": 0.75
    }
  },
  "errors": null
}
```

### Validation error (400)

```json
{
  "ok": false,
  "safezone": null,
  "errors": {
    "zone": "x + width must be <= 1.0."
  }
}
```

Possible error keys include `payload`, `enabled`, `zone`, `x`, `y`, `width`, `height`.

## Request: GET /safezone

### Response (200)

```json
{
  "ok": true,
  "safezone": {
    "enabled": true,
    "zone": {
      "x": 0.15,
      "y": 0.1,
      "width": 0.55,
      "height": 0.75
    }
  },
  "errors": null
}
```

## Alert Event (Socket.IO)

When any detected person leaves the configured safezone, server emits on `alert` channel:

```json
{
  "type": "safezone_exit",
  "message": "Person left safezone!",
  "timestamp": 1740412345.123
}
```

Notes:
- `timestamp` is Unix epoch seconds (`float`)
- Safezone alerts are transition-based (inside → outside), not continuous while staying outside
- Safezone alerts also reuse server cooldown behavior as an additional rate limit
- Current behavior is "any detected person" (not per-person identity tracking)
