"""
Integration utilities for the YOLO and DeepSORT tracker
"""

import cv2
import time
import os
import numpy as np
from pathlib import Path


def download_models():
    """
    Download required YOLO and DeepSORT models if they don't exist
    """
    # Create models directory if it doesn't exist
    os.makedirs("models/weights", exist_ok=True)

    # YOLO model
    yolo_model_path = "models/weights/yolov5s.pt"
    if not os.path.exists(yolo_model_path):
        try:
            # Use torch hub to download YOLOv5s
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            torch.save(model.state_dict(), yolo_model_path)
            print(f"Downloaded YOLOv5 model to {yolo_model_path}")
        except Exception as e:
            print(f"Could not download YOLOv5 model: {e}")

    # DeepSORT model
    deepsort_model_dir = "models/deep_sort_weights"
    deepsort_model_path = f"{deepsort_model_dir}/mars-small128.pb"
    if not os.path.exists(deepsort_model_path):
        os.makedirs(deepsort_model_dir, exist_ok=True)
        try:
            # Download from official source using requests
            import requests
            url = "https://github.com/ZQPei/deep_sort_pytorch/raw/master/deep_sort/deep/checkpoint/mars-small128.pb"
            print(f"Downloading DeepSORT model from {url}")
            r = requests.get(url)
            with open(deepsort_model_path, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded DeepSORT model to {deepsort_model_path}")
        except Exception as e:
            print(f"Could not download DeepSORT model: {e}")

    return yolo_model_path, deepsort_model_path


import cv2
import numpy as np


def initialize_tracker(confidence_threshold=0.5, use_cuda=False):
    """Initialize the DeepSORT tracker with YOLO detector"""
    try:
        # Try to import YOLOv8 with Ultralytics
        try:
            from ultralytics import YOLO

            # Load YOLO model
            model = YOLO('yolov8n.pt')
            print("YOLO model loaded successfully")

            # Import DeepSORT
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort

                # Create tracker instance - removing the use_cuda parameter
                tracker = DeepSort(
                    max_age=30,
                    n_init=2,
                    nms_max_overlap=0.5,
                    max_cosine_distance=0.3,
                    nn_budget=100
                )

                print("DeepSORT tracker initialized")

                # Create a wrapper object that contains both detector and tracker
                class YOLODeepSORTWrapper:
                    def __init__(self, yolo_model, deepsort_tracker, confidence_threshold):
                        self.model = yolo_model
                        self.tracker = deepsort_tracker
                        self.confidence_threshold = confidence_threshold
                        self.classes = {
                            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'
                        }
                        self.vehicle_classes = [2, 5, 7]  # car, bus, truck
                        self.count = 0

                    def __call__(self, frame):
                        """Process frame with YOLO and DeepSORT"""
                        # Run YOLO detection
                        results = self.model(frame, verbose=False)

                        # Extract detections for DeepSORT
                        detections = []
                        boxes = []
                        confidence_scores = []
                        class_ids = []

                        for result in results:
                            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                                for box in result.boxes:
                                    # Get coordinates, confidence and class
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    confidence = float(box.conf)
                                    class_id = int(box.cls)

                                    # Filter out non-vehicles and low confidence
                                    if confidence >= self.confidence_threshold and class_id in self.vehicle_classes:
                                        boxes.append([int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)])
                                        confidence_scores.append(confidence)
                                        class_ids.append(class_id)

                                        # Add to detections list for DeepSORT
                                        detections.append(([int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)],
                                                           confidence,
                                                           class_id))

                        # Update tracker
                        tracks = self.tracker.update_tracks(detections, frame=frame)

                        return tracks, boxes, confidence_scores, class_ids

                    def reset_count(self):
                        """Reset vehicle counter"""
                        self.count = 0

                    def set_confidence_threshold(self, threshold):
                        """Update the confidence threshold"""
                        self.confidence_threshold = threshold
                        print(f"Updated confidence threshold to {threshold}")

                # Create and return the wrapper
                return YOLODeepSORTWrapper(model, tracker, confidence_threshold)

            except ImportError as e:
                print(f"Could not import DeepSORT: {e}")
                return None

        except ImportError as e:
            print(f"Could not import YOLO: {e}")
            return None

    except Exception as e:
        print(f"Error initializing tracker: {e}")
        return None


def process_ml_detections_with_tracking(frame, tracker, line_height, offset, vehicle_counter, classes):
    """Process a frame using YOLO+DeepSORT tracking"""
    if tracker is None:
        # Draw line if no tracker available
        cv2.line(frame, (0, line_height), (frame.shape[1], line_height), (0, 255, 0), 2)
        return frame, [], vehicle_counter

    try:
        # Get tracking results
        tracks, _, _, _ = tracker(frame)

        # Draw detection line
        cv2.line(frame, (0, line_height), (frame.shape[1], line_height), (0, 255, 0), 2)

        vehicle_ids_crossed = []

        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue

            # Get track info
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Calculate center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Check if the vehicle crosses the line
            if not hasattr(track, 'previous_cy'):
                track.previous_cy = None

            if track.previous_cy is not None:
                # If the center crosses the line from top to bottom
                if (track.previous_cy < line_height - offset and cy >= line_height - offset and
                        cy <= line_height + offset and track_id not in vehicle_ids_crossed):
                    vehicle_counter += 1
                    vehicle_ids_crossed.append(track_id)

                    # Draw a filled rectangle to indicate counting
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Store current position for next iteration
            track.previous_cy = cy

        # Draw counter
        cv2.putText(frame, f"Vehicle Count: {vehicle_counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        return frame, vehicle_ids_crossed, vehicle_counter

    except Exception as e:
        print(f"Error in tracking: {e}")
        cv2.putText(frame, f"Tracking Error: {str(e)[:30]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        return frame, [], vehicle_counter