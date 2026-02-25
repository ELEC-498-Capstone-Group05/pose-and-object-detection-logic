import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

import cv2


class RollingVideoRecorder:
    def __init__(
        self,
        output_dir: str = "recordings",
        clip_seconds: int = 60,
        max_storage_gb: float = 80.0,
    ):
        self.output_dir = output_dir
        self.clip_seconds = max(5, int(clip_seconds))
        self.max_storage_bytes = int(max_storage_gb * 1024 * 1024 * 1024)

        self._lock = threading.Lock()
        self._writer = None
        self._width = None
        self._height = None
        self._fps = None
        self._clip_start_ts = 0.0
        self._current_path = None
        self._current_ext = None
        self._frame_counter = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def start(self, width: int, height: int, fps: float) -> None:
        with self._lock:
            self._width = int(width)
            self._height = int(height)
            self._fps = float(fps) if fps and fps > 0 else 30.0
            self._open_new_writer_locked(time.time())
            self._prune_oldest_locked()

    def write(self, frame) -> None:
        with self._lock:
            if self._width is None or self._height is None:
                return

            now = time.time()
            if self._writer is None:
                self._open_new_writer_locked(now)
            elif now - self._clip_start_ts >= self.clip_seconds:
                self._rotate_locked(now)

            if self._writer is not None:
                self._writer.write(frame)
                self._frame_counter += 1

            if self._frame_counter % 300 == 0:
                self._prune_oldest_locked()

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self._current_path = None
            self._current_ext = None

    def list_clips(self) -> List[Dict]:
        with self._lock:
            clips = []
            for name in os.listdir(self.output_dir):
                path = os.path.join(self.output_dir, name)
                if not os.path.isfile(path):
                    continue
                if not self._is_clip_file(name):
                    continue
                st = os.stat(path)
                clips.append(
                    {
                        "filename": name,
                        "size_bytes": st.st_size,
                        "created_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        "modified_ts": st.st_mtime,
                    }
                )

            clips.sort(key=lambda item: item["modified_ts"], reverse=True)
            for item in clips:
                item.pop("modified_ts", None)
            return clips

    def get_clip_path(self, filename: str) -> Optional[str]:
        safe_name = os.path.basename(filename)
        if safe_name != filename:
            return None
        if not self._is_clip_file(safe_name):
            return None

        path = os.path.join(self.output_dir, safe_name)
        if not os.path.isfile(path):
            return None
        return path

    def get_storage_info(self) -> Dict:
        with self._lock:
            used = self._calculate_storage_bytes_locked()
            clips = self._list_clip_paths_locked()
            oldest = None
            newest = None
            if clips:
                oldest = datetime.fromtimestamp(os.path.getmtime(clips[0])).isoformat()
                newest = datetime.fromtimestamp(os.path.getmtime(clips[-1])).isoformat()

            return {
                "enabled": True,
                "output_dir": self.output_dir,
                "clip_seconds": self.clip_seconds,
                "used_bytes": used,
                "max_bytes": self.max_storage_bytes,
                "clip_count": len(clips),
                "oldest_clip": oldest,
                "newest_clip": newest,
                "active_clip": os.path.basename(self._current_path) if self._current_path else None,
            }

    def _rotate_locked(self, now: float) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._current_path = None
        self._current_ext = None
        self._open_new_writer_locked(now)
        self._prune_oldest_locked()

    def _open_new_writer_locked(self, now: float) -> None:
        timestamp = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")

        candidates = [
            ("mp4v", "mp4"),
            ("XVID", "avi"),
            ("MJPG", "avi"),
        ]

        writer = None
        chosen_path = None
        chosen_ext = None

        for codec, ext in candidates:
            path = os.path.join(self.output_dir, f"clip_{timestamp}.{ext}")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(path, fourcc, self._fps, (self._width, self._height))
            if test_writer is not None and test_writer.isOpened():
                writer = test_writer
                chosen_path = path
                chosen_ext = ext
                break
            if test_writer is not None:
                test_writer.release()
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        self._writer = writer
        self._current_path = chosen_path
        self._current_ext = chosen_ext
        self._clip_start_ts = now

    def _prune_oldest_locked(self) -> None:
        total = self._calculate_storage_bytes_locked()
        if total <= self.max_storage_bytes:
            return

        clip_paths = self._list_clip_paths_locked()
        for path in clip_paths:
            if total <= self.max_storage_bytes:
                break
            if self._current_path and os.path.abspath(path) == os.path.abspath(self._current_path):
                continue

            try:
                size = os.path.getsize(path)
                os.remove(path)
                total -= size
            except OSError:
                continue

    def _list_clip_paths_locked(self) -> List[str]:
        paths = []
        for name in os.listdir(self.output_dir):
            if not self._is_clip_file(name):
                continue
            path = os.path.join(self.output_dir, name)
            if os.path.isfile(path):
                paths.append(path)

        paths.sort(key=lambda p: os.path.getmtime(p))
        return paths

    def _calculate_storage_bytes_locked(self) -> int:
        total = 0
        for path in self._list_clip_paths_locked():
            try:
                total += os.path.getsize(path)
            except OSError:
                continue
        return total

    @staticmethod
    def _is_clip_file(name: str) -> bool:
        lower = name.lower()
        return lower.endswith(".mp4") or lower.endswith(".avi")
