"""
Object Detector for YOLO

This module encapsulates YOLO inference, preprocessing, and postprocessing.
It is designed to be used by the main server to run object detection on Edge TPU.
"""

import time
import logging
from pathlib import Path
import numpy as np
import cv2
from pycoral.adapters import common


class TrackedObject:
    """Represents a single object being tracked over time."""
    def __init__(self, box, class_id, score, frame_id, min_hits=3):
        self.box = np.array(box, dtype=np.float32)  # [cx, cy, w, h]
        self.class_id = class_id
        self.score = score
        self.start_frame = frame_id
        self.last_seen_frame = frame_id
        self.hits = 1
        self.min_hits = min_hits
        self.active = False

    def update(self, box, score, frame_id, alpha=0.6):
        """Update the object state with a new detection."""
        self.box = alpha * np.array(box, dtype=np.float32) + (1 - alpha) * self.box
        self.score = alpha * score + (1 - alpha) * self.score
        self.last_seen_frame = frame_id
        self.hits += 1
        if self.hits >= self.min_hits:
            self.active = True


class ObjectTracker:
    """
    Maintains temporal consistency of detections.
    Uses IoU matching and EMA smoothing.
    """
    def __init__(self, iou_threshold=0.1, max_missed_frames=3, min_hits=2, smooth_factor=0.9):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_hits = min_hits
        self.smooth_factor = smooth_factor
        self.frame_count = 0

    def update(self, boxes, class_ids, scores):
        """
        Update tracks with new detections.
        
        Args:
            boxes: numpy array of [cx, cy, w, h]
            class_ids: numpy array of class indices
            scores: numpy array of confidence scores
            
        Returns:
            tuple: (tracked_boxes, tracked_class_ids, tracked_scores)
        """
        self.frame_count += 1
        
        # If no detections, just update missed frames
        if len(boxes) == 0:
            self._prune_tracks()
            return self._get_active_tracks()

        # Match existing tracks to new detections
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(boxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(boxes)))
        for t, track in enumerate(self.tracks):
             for d, box in enumerate(boxes):
                 if track.class_id == class_ids[d]:
                     iou_matrix[t, d] = self._calculate_iou(track.box, box)
        
        # Greedy matching
        if len(self.tracks) > 0 and len(boxes) > 0:
            while True:
                if iou_matrix.size == 0 or np.all(iou_matrix == -1):
                    break
                    
                ind = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
                max_iou = iou_matrix[ind]
                
                if max_iou < self.iou_threshold:
                    break
                    
                t_idx, d_idx = ind
                
                # Update track
                self.tracks[t_idx].update(
                    boxes[d_idx], 
                    scores[d_idx], 
                    self.frame_count, 
                    self.smooth_factor
                )
                
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
                    
                # Mark as processed
                iou_matrix[t_idx, :] = -1
                iou_matrix[:, d_idx] = -1

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            new_track = TrackedObject(
                boxes[d_idx], 
                class_ids[d_idx], 
                scores[d_idx], 
                self.frame_count,
                self.min_hits
            )
            self.tracks.append(new_track)

        self._prune_tracks()
        return self._get_active_tracks()

    def _prune_tracks(self):
        self.tracks = [
            t for t in self.tracks 
            if (self.frame_count - t.last_seen_frame) <= self.max_missed_frames
        ]

    def _get_active_tracks(self):
        active = [t for t in self.tracks if t.active]
        if not active:
            return np.array([]), np.array([]), np.array([])
            
        boxes = np.array([t.box for t in active])
        class_ids = np.array([t.class_id for t in active])
        scores = np.array([t.score for t in active])
        return boxes, class_ids, scores

    def _calculate_iou(self, box1, box2):
        # box: [cx, cy, w, h]
        b1_x1 = box1[0] - box1[2] / 2
        b1_y1 = box1[1] - box1[3] / 2
        b1_x2 = box1[0] + box1[2] / 2
        b1_y2 = box1[1] + box1[3] / 2
        
        b2_x1 = box2[0] - box2[2] / 2
        b2_y1 = box2[1] - box2[3] / 2
        b2_x2 = box2[0] + box2[2] / 2
        b2_y2 = box2[1] + box2[3] / 2
        
        xx1 = max(b1_x1, b2_x1)
        yy1 = max(b1_y1, b2_y1)
        xx2 = min(b1_x2, b2_x2)
        yy2 = min(b1_y2, b2_y2)
        
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter = w * h
        
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        
        union = area1 + area2 - inter
        if union <= 0: return 0
        return inter / union


class ObjectDetector:
    """Encapsulates YOLO object detection logic."""

    def __init__(
        self,
        interpreter,
        labels=None,
        conf_threshold=0.30,
        iou_threshold=0.45,
        debug=False,
        debug_dir="debug_yolo",
        debug_every=60,
    ):
        self.interpreter = interpreter
        self.labels = labels or {}
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self.debug_every = max(1, int(debug_every))
        self._frame_count = 0
        self._log = logging.getLogger(__name__)
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.input_size = common.input_size(self.interpreter)
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        self.input_scale = input_details.get('quantization', (1.0, 0))[0]
        self.input_zero_point = input_details.get('quantization', (1.0, 0))[1]
        self.output_scale = output_details.get('quantization', (1.0, 0))[0]
        self.output_zero_point = output_details.get('quantization', (1.0, 0))[1]

    @staticmethod
    def _non_max_suppression(boxes, scores, threshold):
        """Performs Non-Maximum Suppression (NMS) on the boxes."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    def _letterbox(self, rgb_frame):
        """Resize with padding to preserve aspect ratio.

        Returns: resized, scale, pad_left, pad_top
        """
        in_w, in_h = self.input_size
        h, w, _ = rgb_frame.shape
        scale = min(in_w / w, in_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_x = in_w - new_w
        pad_y = in_h - new_h
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, pad_left, pad_top

    def _process_yolo_output(self, output):
        """Processes YOLO output tensor."""
        output = output.transpose((0, 2, 1))
        pred = output[0]

        boxes = pred[:, :4]
        scores = pred[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        mask = max_scores > self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        max_scores = max_scores[mask]

        if len(boxes) == 0:
            return [], [], []

        keep = self._non_max_suppression(boxes, max_scores, self.iou_threshold)
        return boxes[keep], class_ids[keep], max_scores[keep]

    def infer(self, frame_bgr):
        """
        Run YOLO inference on a BGR frame.

        Returns:
            tuple: (boxes, class_ids, scores, timings)
        """
        t0 = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized, scale, pad_left, pad_top = self._letterbox(rgb_frame)

        input_data = resized.astype(np.float32) / 255.0
        input_data = (input_data / self.input_scale) + self.input_zero_point
        input_data = np.clip(np.rint(input_data), -128, 127).astype(np.int8)
        should_dump = self.debug and (self._frame_count % self.debug_every == 0)
        if should_dump:
            try:
                in_min = int(input_data.min())
                in_max = int(input_data.max())
                in_mean = float(input_data.mean())
                self._log.info(
                    "YOLO input stats (int8): min=%d max=%d mean=%.2f scale=%.6f zero=%d size=%s",
                    in_min,
                    in_max,
                    in_mean,
                    self.input_scale,
                    self.input_zero_point,
                    self.input_size,
                )

                deq = (input_data.astype(np.float32) - self.input_zero_point) * self.input_scale
                deq = np.clip(deq * 255.0, 0, 255).astype(np.uint8)
                dbg_bgr = cv2.cvtColor(deq, cv2.COLOR_RGB2BGR)
                out_path = self.debug_dir / f"input_{self._frame_count:06d}.jpg"
                cv2.imwrite(str(out_path), dbg_bgr)
            except Exception as exc:
                self._log.warning("YOLO debug dump failed: %s", exc)
        pre_ms = (time.perf_counter() - t0) * 1000

        common.set_input(self.interpreter, input_data)

        t1 = time.perf_counter()
        self.interpreter.invoke()
        invoke_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        output = common.output_tensor(self.interpreter, 0).copy()
        output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale
        if should_dump:
            try:
                out_min = float(output.min())
                out_max = float(output.max())
                self._log.info(
                    "YOLO output stats (dequant): min=%.6f max=%.6f scale=%.6f zero=%d",
                    out_min,
                    out_max,
                    self.output_scale,
                    self.output_zero_point,
                )
            except Exception as exc:
                self._log.warning("YOLO output stats failed: %s", exc)
        boxes, class_ids, scores = self._process_yolo_output(output)
        self._frame_count += 1
        post_ms = (time.perf_counter() - t2) * 1000

        timings = {
            'pre_ms': pre_ms,
            'invoke_ms': invoke_ms,
            'post_ms': post_ms,
        }

        meta = {
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'input_size': self.input_size,
            'letterbox': True,
        }

        return boxes, class_ids, scores, timings, meta
