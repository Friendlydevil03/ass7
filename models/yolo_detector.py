import cv2
import numpy as np
import torch
import os
import time
from pathlib import Path
from collections import OrderedDict


class YOLODetector:
    """
    YOLO-based detector for vehicle detection
    Supports YOLOv5 and YOLOv8 models
    """

    def __init__(self, model_path=None, confidence_threshold=0.5, use_cuda=True):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

        # Cache for previously detected frames to improve performance
        self.detection_cache = OrderedDict()
        self.cache_max_size = 20
        self.last_inference_time = 0
        self.inference_interval = 0.5  # Minimum time between full inferences (seconds)

        # Classes we're interested in for vehicle detection
        self.vehicle_classes = [2, 3, 5, 6, 7, 8]  # car, motorcycle, bus, train, truck, boat

        # Load model
        self.model = self._load_model(model_path)
        self.model_type = "yolov5"  # Default model type

        print(f"YOLO detector initialized using {self.device}")

    def _load_model(self, model_path=None):
        """Load the YOLO model"""
        try:
            # First try to import yolov5 as a package
            try:
                import yolov5
                print("Using yolov5 package...")

                # If no model path specified, use default YOLOv5s
                if model_path is None or not os.path.exists(model_path):
                    # Need to use a specific format for loading models
                    model = yolov5.load('yolov5s.pt')  # Note the .pt extension!
                else:
                    model = yolov5.load(model_path)

                model.to(self.device)
                return model

            except (ImportError, Exception) as e:
                print(f"Error with yolov5 package: {e}")
                print("Trying alternative approach...")

                # Create a simple wrapper for object detection from CV2
                class SimpleDetector:
                    def __init__(self, confidence_threshold=0.5):
                        # Try to load DNN-based model
                        try:
                            print("Loading CV2 DNN model...")
                            # Load pre-trained MobileNet SSD model
                            self.net = cv2.dnn.readNetFromCaffe(
                                "models/MobileNetSSD_deploy.prototxt",
                                "models/MobileNetSSD_deploy.caffemodel"
                            )
                            self.classes = ["background", "person", "bicycle", "car", "motorcycle",
                                            "airplane", "bus", "train", "truck", "boat"]
                            self.confidence_threshold = confidence_threshold
                            print("CV2 model loaded successfully")
                        except Exception as e:
                            print(f"Failed to load CV2 model: {e}")
                            self.net = None

                    def __call__(self, img):
                        # Simple wrapper to match yolov5 API
                        if self.net is None:
                            # Return empty results if model failed to load
                            return SimpleResults()

                        height, width = img.shape[:2]
                        blob = cv2.dnn.blobFromImage(
                            cv2.resize(img, (300, 300)),
                            0.007843, (300, 300), 127.5
                        )

                        # Run detection
                        self.net.setInput(blob)
                        detections = self.net.forward()

                        # Process detections
                        results = []
                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > self.confidence_threshold:
                                class_id = int(detections[0, 0, i, 1])
                                # Get bounding box
                                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                                (x1, y1, x2, y2) = box.astype("int")
                                # Create result in format similar to yolov5
                                result = [x1, y1, x2, y2, float(confidence), class_id]
                                results.append(result)

                        # Return in format similar to yolov5
                        return SimpleResults(results)

                # Simple class to mimic yolov5 results
                class SimpleResults:
                    def __init__(self, detections=None):
                        self.xyxy = [np.array(detections if detections else [])]

                # Return simple detector
                return SimpleDetector(self.confidence_threshold)

        except Exception as e:
            print(f"Error loading object detection model: {str(e)}")

            # Return a dummy model that won't crash the app
            class DummyModel:
                def __call__(self, img):
                    print("WARNING: Using dummy detection model")
                    return SimpleResults()

                def to(self, device):
                    return self

            return DummyModel()

    def detect_vehicles(self, frame):
        """Detect vehicles in a frame with caching and optimization"""
        # Handle invalid input or no model
        if frame is None or frame.size == 0 or self.model is None:
            return []

        try:
            # Generate a frame hash for cache lookup
            small_frame = cv2.resize(frame, (32, 32))
            frame_hash = hash(small_frame.tobytes())

            # Check if we have this frame in cache
            if frame_hash in self.detection_cache:
                return self.detection_cache[frame_hash]

            # Check if we should do a full inference based on time
            current_time = time.time()
            if current_time - self.last_inference_time < self.inference_interval:
                # Return the most recent detection if available
                if self.detection_cache:
                    return list(self.detection_cache.values())[-1]

            # Update last inference time
            self.last_inference_time = current_time

            vehicle_detections = []

            # Handle different model types
            if self.model_type == "fasterrcnn":
                # Convert frame to tensor efficiently
                img = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
                img = img.to(self.device)
                # Create a list of tensors as expected by the model
                img_list = [img[0]]

                with torch.no_grad():
                    predictions = self.model(img_list)

                # Extract detections
                boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
                scores = predictions[0]['scores'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()

                # Filter by confidence and vehicle classes
                for box, score, label in zip(boxes, scores, labels):
                    if score > self.confidence_threshold and label in self.vehicle_classes:
                        vehicle_detections.append(([box[0], box[1], box[2], box[3]], float(score), int(label)))

            elif self.model_type == "yolov5":
                # Use YOLOv5 API
                results = self.model(frame)

                # Extract detections from results
                for det in results.xyxy[0]:  # Process first image in batch
                    if len(det) >= 6:  # Make sure we have all elements
                        x1, y1, x2, y2, conf, cls = det[:6]

                        # Convert tensor values to Python types
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = float(conf)
                        cls = int(cls)

                        if conf > self.confidence_threshold and cls in self.vehicle_classes:
                            vehicle_detections.append(
                                ([x1, y1, x2, y2], conf, cls)
                            )

            elif self.model_type == "opencv":
                # Use OpenCV DNN model
                height, width = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)),
                    0.007843, (300, 300), 127.5
                )

                # Run detection
                self.model.setInput(blob)
                detections = self.model.forward()

                # Process detections
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.confidence_threshold:
                        class_id = int(detections[0, 0, i, 1])
                        if class_id in self.vehicle_classes:
                            # Get bounding box
                            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                            (x1, y1, x2, y2) = box.astype("int")
                            vehicle_detections.append(
                                ([x1, y1, x2, y2], float(confidence), class_id)
                            )

            # Store in cache
            self.detection_cache[frame_hash] = vehicle_detections

            # Limit cache size
            while len(self.detection_cache) > self.cache_max_size:
                self.detection_cache.popitem(last=False)  # Remove oldest item (first=False means FIFO)

            return vehicle_detections

        except Exception as e:
            print(f"Error in vehicle detection: {str(e)}")
            # Return empty list on error to prevent crashing
            return []

    def set_confidence_threshold(self, threshold):
        """Update the confidence threshold"""
        self.confidence_threshold = threshold
        # Clear cache when threshold changes
        self.detection_cache.clear()