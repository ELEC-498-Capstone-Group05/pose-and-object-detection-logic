# Mobile App Integration: Video Download Endpoint

This document explains exactly how to parse and use the video clip endpoints exposed by the Pi server.

## Base URL

Use the Pi IP and port where `server.py` is running:

- `http://<PI_IP>:5000`
- Example: `http://10.216.148.225:5000`

---

## 1) List available clips

### Endpoint

- **Method:** `GET`
- **URL:** `/clips`
- **Full Example:** `GET http://<PI_IP>:5000/clips`

### Response shape (exact)

```json
{
  "clips": [
    {
      "filename": "clip_20260225_153000.mp4",
      "size_bytes": 18765432,
      "created_at": "2026-02-25T15:30:59.123456"
    }
  ],
  "storage": {
    "enabled": true,
    "output_dir": "recordings",
    "clip_seconds": 60,
    "used_bytes": 4287643210,
    "max_bytes": 85899345920,
    "clip_count": 240,
    "oldest_clip": "2026-02-25T11:20:00.000000",
    "newest_clip": "2026-02-25T15:30:59.123456",
    "active_clip": "clip_20260225_153100.mp4"
  }
}
```

### Notes

- `clips` is sorted **newest first**.
- `filename` is the ID you use for downloading.
- `size_bytes` is file size in bytes.
- `storage.max_bytes` corresponds to your configured cap (80 GB by default).

---

## 2) Download a specific clip

### Endpoint

- **Method:** `GET`
- **URL:** `/clips/<filename>`
- **Full Example:** `GET http://<PI_IP>:5000/clips/clip_20260225_153000.mp4`

### Behavior

- Returns the binary video file as an attachment.
- If file is missing/invalid, returns **404**.

---

## TypeScript parsing contracts (mobile app)

```ts
export type ClipItem = {
  filename: string;
  size_bytes: number;
  created_at: string;
};

export type StorageInfo = {
  enabled: boolean;
  output_dir?: string;
  clip_seconds?: number;
  used_bytes?: number;
  max_bytes?: number;
  clip_count?: number;
  oldest_clip?: string | null;
  newest_clip?: string | null;
  active_clip?: string | null;
};

export type ClipsResponse = {
  clips: ClipItem[];
  storage: StorageInfo;
};
```

---

## React Native usage (fetch + parse)

```ts
const BASE_URL = "http://<PI_IP>:5000";

export async function fetchClips(): Promise<ClipsResponse> {
  const res = await fetch(`${BASE_URL}/clips`);
  if (!res.ok) {
    throw new Error(`Failed to fetch clips: ${res.status}`);
  }

  const json = (await res.json()) as ClipsResponse;

  if (!Array.isArray(json.clips) || typeof json.storage !== "object") {
    throw new Error("Invalid /clips payload");
  }

  return json;
}
```

---

## Download to device (Expo)

If your app uses Expo, use `expo-file-system`:

```ts
import * as FileSystem from "expo-file-system";

const BASE_URL = "http://<PI_IP>:5000";

export async function downloadClipExpo(filename: string): Promise<string> {
  const encoded = encodeURIComponent(filename);
  const url = `${BASE_URL}/clips/${encoded}`;

  const targetPath = `${FileSystem.documentDirectory}${filename}`;
  const result = await FileSystem.downloadAsync(url, targetPath);

  if (result.status !== 200) {
    throw new Error(`Download failed: ${result.status}`);
  }

  return result.uri;
}
```

---

## Download to device (Bare React Native)

If your app is not Expo, use `react-native-fs`:

```ts
import RNFS from "react-native-fs";

const BASE_URL = "http://<PI_IP>:5000";

export async function downloadClipRNFS(filename: string): Promise<string> {
  const encoded = encodeURIComponent(filename);
  const fromUrl = `${BASE_URL}/clips/${encoded}`;
  const toFile = `${RNFS.DocumentDirectoryPath}/${filename}`;

  const result = await RNFS.downloadFile({ fromUrl, toFile }).promise;
  if (result.statusCode !== 200) {
    throw new Error(`Download failed: ${result.statusCode}`);
  }

  return toFile;
}
```

---

## Suggested app flow

1. Call `GET /clips`.
2. Render `clips` list (newest first).
3. On user tap, call `downloadClip...` with `clip.filename`.
4. Save local URI/path in app state for playback or share.
5. Handle `404` by refreshing the list (file may have been auto-deleted by retention).

---

## Error handling checklist

- `GET /clips` non-200: show server unavailable/retry UI.
- Download 404: show “Clip expired or deleted” and refresh list.
- Disk full on phone: catch filesystem write errors and prompt user.
- Network timeout: retry with backoff and allow cancel.

---

## Quick manual test

```bash
# 1) List clips
curl http://<PI_IP>:5000/clips

# 2) Download one clip
curl -L -o test_clip.mp4 "http://<PI_IP>:5000/clips/clip_20260225_153000.mp4"
```

If `test_clip.mp4` plays, the endpoint is working end-to-end for mobile download.
