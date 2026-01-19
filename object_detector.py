"""
Object Detector for YOLO

This module encapsulates YOLO inference, preprocessing, and postprocessing.
It is designed to be used by the main server to run object detection on Edge TPU.
"""

import time
import numpy as np
import cv2
from pycoral.adapters import common


class ObjectDetector:
    """Encapsulates YOLO object detection logic."""

    def __init__(self, interpreter, labels=None, conf_threshold=0.40, iou_threshold=0.45):
        self.interpreter = interpreter
        self.labels = labels or {}
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

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
        resized = cv2.resize(rgb_frame, self.input_size, interpolation=cv2.INTER_NEAREST)

        input_data = resized.astype(np.float32) / 255.0
        input_data = (input_data / self.input_scale) + self.input_zero_point
        input_data = np.clip(np.rint(input_data), -128, 127).astype(np.int8)
        pre_ms = (time.perf_counter() - t0) * 1000

        common.set_input(self.interpreter, input_data)

        t1 = time.perf_counter()
        self.interpreter.invoke()
        invoke_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        output = common.output_tensor(self.interpreter, 0).copy()
        output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale
        boxes, class_ids, scores = self._process_yolo_output(output)
        post_ms = (time.perf_counter() - t2) * 1000

        timings = {
            'pre_ms': pre_ms,
            'invoke_ms': invoke_ms,
            'post_ms': post_ms,
        }

        return boxes, class_ids, scores, timings
