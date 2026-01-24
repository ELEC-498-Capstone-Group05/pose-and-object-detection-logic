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


class ObjectDetector:
    """Encapsulates YOLO object detection logic."""

    def __init__(
        self,
        interpreter,
        labels=None,
        conf_threshold=0.40,
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
