import cv2
import numpy as np
import time
from models.yolo_detector import YOLODetector
from models.deep_sort_tracker import DeepSORTTracker


class VehicleTracker:
    """
    Combined vehicle detection and tracking using YOLO and DeepSORT
    """

    def __init__(self, yolo_model_path=None, deepsort_model_path=None,
                 confidence_threshold=0.5, use_cuda=True):
        self.confidence_threshold = confidence_threshold

        # Initialize YOLO detector
        self.detector = YOLODetector(
            model_path=yolo_model_path,
            confidence_threshold=confidence_threshold,
            use_cuda=use_cuda
        )

        # Initialize DeepSORT tracker
        self.tracker = DeepSORTTracker(
            model_path=deepsort_model_path,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )

        # For performance tracking
        self.detection_time = 0
        self.tracking_time = 0
        self.total_time = 0
        self.frame_count = 0

        # Class names for display
        self.classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train', 'truck', 'boat'
        ]

        # Tracking history
        self.tracked_vehicles = {}  # Dictionary to store vehicle data by ID
        self.vehicle_count = 0

        print("Vehicle tracker initialized with YOLO and DeepSORT")

    def process_frame(self, frame, line_height=None, offset=10):
        """
        Process a frame for vehicle detection and tracking

        Args:
            frame: Input frame
            line_height: Y-coordinate of counting line (optional)
            offset: Offset for counting line (optional)

        Returns:
            tuple: (processed_frame, detections, tracks, vehicle_count)
        """
        if frame is None or frame.size == 0:
            print("Error: Empty frame received")
            return None, [], [], self.vehicle_count

        # Make a copy for drawing
        processed_frame = frame.copy()
        start_time = time.time()

        try:
            # Step 1: Run YOLO detector
            detection_start = time.time()
            detections = self.detector.detect_vehicles(frame)
            self.detection_time = time.time() - detection_start

            # Step 2: Update tracker with detections
            tracking_start = time.time()
            tracks = self.tracker.update(frame, detections)
            self.tracking_time = time.time() - tracking_start

            # Step 3: Process tracks, count vehicles crossing line
            if line_height is not None:
                processed_frame, self.vehicle_count = self._process_tracks(
                    processed_frame, tracks, line_height, offset)
            else:
                # Just draw the tracks without counting
                processed_frame = self.tracker.draw_tracks(processed_frame, tracks)

            # Update performance metrics
            self.total_time = time.time() - start_time
            self.frame_count += 1

            # Draw performance stats
            self._draw_stats(processed_frame)

            return processed_frame, detections, tracks, self.vehicle_count

        except Exception as e:
            print(f"Error in vehicle tracking: {str(e)}")
            self.total_time = time.time() - start_time
            return frame, [], [], self.vehicle_count

    def _process_tracks(self, frame, tracks, line_height, offset):
        """
        Process tracks and count vehicles crossing a line

        Args:
            frame: Frame to draw on
            tracks: List of tracks (ID, bbox, class_id)
            line_height: Y-coordinate of counting line
            offset: Offset for counting line

        Returns:
            tuple: (processed_frame, vehicle_count)
        """
        # Draw counting line
        cv2.line(frame, (0, line_height), (frame.shape[1], line_height), (0, 255, 0), 2)

        vehicle_count = self.vehicle_count

        # Process each track
        for track_id, bbox, class_id in tracks:
            # Calculate centroid
            x1, y1, x2, y2 = bbox
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2

            # Check if this is a new track
            if track_id not in self.tracked_vehicles:
                self.tracked_vehicles[track_id] = {
                    'positions': [(centroid_x, centroid_y)],
                    'counted': False,
                    'class_id': class_id,
                    'first_seen': time.time()
                }
            else:
                # Add new position to track history
                self.tracked_vehicles[track_id]['positions'].append((centroid_x, centroid_y))

                # Keep a limited history to save memory
                if len(self.tracked_vehicles[track_id]['positions']) > 30:
                    self.tracked_vehicles[track_id]['positions'] = self.tracked_vehicles[track_id]['positions'][-30:]

            # Check if vehicle has crossed the line
            if not self.tracked_vehicles[track_id]['counted']:
                # We need at least 2 positions to check crossing
                positions = self.tracked_vehicles[track_id]['positions']
                if len(positions) >= 2:
                    prev_y = positions[-2][1]
                    curr_y = positions[-1][1]

                    # Check if the vehicle crossed the line from top to bottom
                    if prev_y < line_height - offset and curr_y > line_height + offset:
                        vehicle_count += 1
                        self.tracked_vehicles[track_id]['counted'] = True

                        # Draw a highlight for counted vehicles
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.putText(frame, f"ID:{track_id} COUNTED", (x1, y1 - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw the tracks
        frame = self.tracker.draw_tracks(frame, tracks)

        # Draw vehicle count
        cv2.putText(frame, f"Total Vehicle Count: {vehicle_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)

        return frame, vehicle_count

    def _draw_stats(self, frame):
        """Draw performance statistics on frame"""
        # Only show stats after processing some frames
        if self.frame_count < 10:
            return frame

        # Calculate average times
        avg_detection = self.detection_time * 1000  # ms
        avg_tracking = self.tracking_time * 1000  # ms
        avg_total = self.total_time * 1000  # ms
        fps = 1.0 / self.total_time if self.total_time > 0 else 0

        # Draw stats
        cv2.putText(frame, f"Detection: {avg_detection:.1f}ms",
                    (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracking: {avg_tracking:.1f}ms",
                    (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {avg_total:.1f}ms, FPS: {fps:.1f}",
                    (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def reset_count(self):
        """Reset the vehicle count"""
        self.vehicle_count = 0
        # Clear tracking history
        self.tracked_vehicles = {}

    def set_confidence_threshold(self, threshold):
        """Update the confidence threshold"""
        self.confidence_threshold = threshold
        self.detector.set_confidence_threshold(threshold)