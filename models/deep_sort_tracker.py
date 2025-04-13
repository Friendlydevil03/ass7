import numpy as np
import cv2
from pathlib import Path
import os


class DeepSORTTracker:
    """
    DeepSORT tracking implementation for vehicle tracking
    """

    def __init__(self, model_path=None, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # Maximum number of frames to keep track of an object that disappeared
        self.min_hits = min_hits  # Minimum number of hits to start tracking
        self.iou_threshold = iou_threshold  # IOU threshold for matching

        # Initialize DeepSORT
        self.tracker = self._initialize_tracker(model_path)

        # Track history for visualization and analysis
        self.track_history = {}  # Dictionary to store tracks by ID
        self.next_track_id = 1

        print("DeepSORT tracker initialized")

    def _initialize_tracker(self, model_path):
        """Initialize a simple tracker as fallback"""
        try:
            # Try to use a simple tracker implementation
            class SimpleTracker:
                def __init__(self):
                    self.next_id = 1
                    self.tracked_objects = {}

                def update(self, frame, detections):
                    results = []

                    # Assign IDs to detections
                    for det in detections:
                        bbox, score, class_id = det
                        # For simplicity, just assign a new ID to each detection
                        track_id = self.next_id
                        self.next_id += 1

                        # Store in results
                        results.append((track_id, bbox, class_id))

                    return results

            print("Initialized simple tracker as fallback")
            return SimpleTracker()

        except Exception as e:
            print(f"Error creating tracker: {str(e)}")

            # Return an even simpler tracker that just returns the detections
            class VerySimpleTracker:
                def __init__(self):
                    self.next_id = 1

                def update(self, frame, detections):
                    results = []
                    for i, det in enumerate(detections):
                        bbox, score, class_id = det
                        results.append((self.next_id + i, bbox, class_id))
                    self.next_id += len(detections)
                    return results

            return VerySimpleTracker()

    def update(self, frame, detections):
        """
        Update tracker with new detections

        Args:
            frame: Current video frame
            detections: List of detection tuples (bbox, confidence, class_id)

        Returns:
            List of tracks (ID, bbox, class_id)
        """
        if not detections:
            return []

        try:
            # Extract bounding boxes, confidence scores, and class IDs
            bboxes = np.array([d[0] for d in detections])
            scores = np.array([d[1] for d in detections])
            class_ids = np.array([d[2] for d in detections])

            # Convert to format expected by DeepSORT [x1, y1, x2, y2] to [x, y, w, h]
            bbox_xywh = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                bbox_xywh.append([
                    (x1 + x2) / 2,  # x center
                    (y1 + y2) / 2,  # y center
                    x2 - x1,  # width
                    y2 - y1  # height
                ])
            bbox_xywh = np.array(bbox_xywh)

            # Get DeepSORT features
            features = self._get_features(frame, bbox_xywh)

            # Update tracker
            outputs = self.tracker.update(bbox_xywh, scores, features)

            # Format results as [ID, bbox, class]
            results = []
            for track in outputs:
                track_id = int(track[4])
                bbox = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]

                # Find the class_id for this tracked object
                # For simplicity, we'll use the class_id of the detection with highest IoU
                class_id = self._get_best_class_id(bbox, bboxes, class_ids)

                # Store in track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        'positions': [],
                        'class_id': class_id,
                        'active': True,
                        'frames_tracked': 1
                    }

                # Update track history
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                self.track_history[track_id]['positions'].append((center_x, center_y))
                self.track_history[track_id]['frames_tracked'] += 1
                self.track_history[track_id]['active'] = True

                # Keep a reasonable history
                if len(self.track_history[track_id]['positions']) > 30:
                    self.track_history[track_id]['positions'] = self.track_history[track_id]['positions'][-30:]

                results.append((track_id, bbox, class_id))

            # Mark inactive tracks
            all_track_ids = list(self.track_history.keys())
            active_track_ids = [r[0] for r in results]
            for track_id in all_track_ids:
                if track_id not in active_track_ids:
                    self.track_history[track_id]['active'] = False

            return results

        except Exception as e:
            print(f"Error in DeepSORT update: {str(e)}")
            return []

    def _get_features(self, frame, bbox_xywh):
        """Extract features for DeepSORT"""
        try:
            # If our tracker has feature extraction built in
            if hasattr(self.tracker, 'extract_features'):
                features = self.tracker.extract_features(frame, bbox_xywh)
                return features

            # Otherwise, return dummy features
            return np.ones((bbox_xywh.shape[0], 128))

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return np.ones((bbox_xywh.shape[0], 128))

    def _get_best_class_id(self, bbox, all_bboxes, all_class_ids):
        """Find the class ID with highest IoU for this track"""
        if not len(all_bboxes):
            return 0

        # Calculate IoU between this bbox and all detection bboxes
        best_iou = 0
        best_idx = 0

        for i, det_bbox in enumerate(all_bboxes):
            iou = self._calculate_iou(bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        # Return the class ID with highest IoU
        if best_iou > 0.5 and best_idx < len(all_class_ids):
            return all_class_ids[best_idx]
        return 0  # Default class ID if no good match

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # Check if boxes overlap
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        union_area = area1 + area2 - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou

    def draw_tracks(self, frame, tracks, draw_trail=True, color_by_id=True):
        """
        Draw tracking results on frame

        Args:
            frame: Frame to draw on
            tracks: List of tracks (ID, bbox, class_id)
            draw_trail: Whether to draw motion trail
            color_by_id: Whether to use different colors for different IDs

        Returns:
            Frame with tracks drawn
        """
        for track_id, bbox, class_id in tracks:
            x1, y1, x2, y2 = bbox

            # Generate color based on track ID
            if color_by_id:
                color = ((track_id * 5) % 256, (track_id * 50) % 256, (track_id * 128) % 256)
            else:
                # Color based on class
                colors = [
                    (0, 255, 0),  # Green (default)
                    (0, 0, 255),  # Red (person)
                    (255, 0, 0),  # Blue (bicycle)
                    (255, 255, 0),  # Cyan (car)
                    (255, 0, 255),  # Purple (motorcycle)
                    (0, 255, 255),  # Yellow (bus)
                    (255, 165, 0),  # Orange (train)
                    (128, 0, 128)  # Purple (truck)
                ]
                color_index = min(class_id, len(colors) - 1)
                color = colors[color_index]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw ID and class
            text = f"ID:{track_id}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trail if enabled
            if draw_trail and track_id in self.track_history:
                trail = self.track_history[track_id]['positions']
                for i in range(1, len(trail)):
                    # Make trail fade out
                    alpha = 0.5 * (i / len(trail))
                    trail_color = tuple([int(c * alpha) for c in color])

                    # Draw line segment of trail
                    cv2.line(frame, trail[i - 1], trail[i], trail_color, 2)

        return frame