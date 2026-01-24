
import unittest
import numpy as np
from object_detector import ObjectTracker

class TestObjectTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracker(iou_threshold=0.3, max_missed_frames=2, min_hits=2, smooth_factor=0.5)

    def test_tracking(self):
        # Frame 1: One object
        boxes1 = np.array([[100, 100, 50, 50]]) # cx, cy, w, h
        classes1 = np.array([0])
        scores1 = np.array([0.9])
        
        # Should return nothing because min_hits=2
        t_boxes, t_classes, t_scores = self.tracker.update(boxes1, classes1, scores1)
        self.assertEqual(len(t_boxes), 0)
        
        # Frame 2: Same object slightly moved
        boxes2 = np.array([[102, 102, 50, 50]])
        classes2 = np.array([0])
        scores2 = np.array([0.9])
        
        # Should return the object now (hits=2)
        t_boxes, t_classes, t_scores = self.tracker.update(boxes2, classes2, scores2)
        self.assertEqual(len(t_boxes), 1)
        
        # Check smoothing: (100*0.5 + 102*0.5) = 101
        self.assertAlmostEqual(t_boxes[0][0], 101.0, places=1)
        
    def test_missed_frames(self):
        # Frame 1 & 2: Detection present
        self.tracker.update(np.array([[100, 100, 50, 50]]), np.array([0]), np.array([0.9]))
        t_boxes, _, _ = self.tracker.update(np.array([[100, 100, 50, 50]]), np.array([0]), np.array([0.9]))
        self.assertEqual(len(t_boxes), 1)
        
        # Frame 3: Missed
        t_boxes, _, _ = self.tracker.update([], [], [])
        # Should still exist because max_missed_frames=2
        self.assertEqual(len(t_boxes), 1)
        
        # Frame 4: Missed again
        t_boxes, _, _ = self.tracker.update([], [], [])
        self.assertEqual(len(t_boxes), 1)
        
        # Frame 5: Missed again
        t_boxes, _, _ = self.tracker.update([], [], [])
        # Should be gone now (missed > 2)
        self.assertEqual(len(t_boxes), 0)

if __name__ == '__main__':
    unittest.main()
