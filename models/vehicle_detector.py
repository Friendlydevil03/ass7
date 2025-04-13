import torch
from torchvision.models import detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import time
import cv2


class VehicleDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold

        # Force CPU usage to avoid CUDA issues
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # Cache for previously detected frames to improve performance
        self.detection_cache = {}
        self.cache_max_size = 20
        self.last_inference_time = 0
        self.inference_interval = 0.5  # Minimum time between full inferences (seconds)

        try:
            # Try to load FasterRCNN model
            print("Loading ML model...")
            try:
                self.model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                # Optimize model for inference
                self.model.eval()
                # Move model to device
                self.model = self.model.to(self.device)
                # Use TorchScript to optimize model if possible
                try:
                    self.model = torch.jit.script(self.model)
                    print("Model optimized with TorchScript")
                except Exception as e:
                    print(f"Could not optimize with TorchScript: {e}")

                print("Model loaded successfully")
                self.model_type = "fasterrcnn"
            except Exception as e:
                print(f"Could not load FasterRCNN: {e}")
                print("Trying to load YOLOv8 model instead...")

                try:
                    # Updated to use ultralytics (YOLOv8) instead of yolov5
                    from ultralytics import YOLO
                    self.model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8 model
                    print("YOLOv8 model loaded successfully")
                    self.model_type = "yolov8"
                except Exception as e:
                    print(f"Could not load YOLOv8: {e}")
                    print("Loading fallback OpenCV DNN model...")

                    # Load OpenCV DNN model as fallback
                    try:
                        # Check if model files exist before attempting to load
                        import os
                        proto_path = "models/MobileNetSSD_deploy.prototxt"
                        model_path = "models/MobileNetSSD_deploy.caffemodel"

                        if not os.path.exists(proto_path):
                            print(f"ERROR: Missing prototxt file at {proto_path}")
                            raise FileNotFoundError(f"Missing {proto_path}")

                        if not os.path.exists(model_path):
                            print(f"ERROR: Missing caffemodel file at {model_path}")
                            raise FileNotFoundError(f"Missing {model_path}")

                        self.model = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                        print("OpenCV DNN model loaded successfully")
                        self.model_type = "opencv"
                    except Exception as e:
                        print(f"Could not load OpenCV DNN model: {e}")
                        print("WARNING: No detection model loaded. Application will run in degraded mode.")
                        self.model = None
                        self.model_type = "none"

            # COCO class names (we're interested in vehicles)
            self.classes = [
                'background', 'person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat'
            ]
            self.vehicle_classes = [2, 3, 5, 6, 7, 8]  # Indices of vehicle classes
        except Exception as e:
            print(f"Error loading ML model: {str(e)}")
            self.model = None
            self.model_type = "none"

    # Add detect_vehicles method that was missing
    def detect_vehicles(self, image):
        """Detect vehicles in an image and return bounding boxes, classes, and scores"""
        if self.model is None:
            return []

        try:
            if self.model_type == "fasterrcnn":
                # Convert image to tensor
                img_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0).to(
                    self.device)

                # Perform inference
                with torch.no_grad():
                    prediction = self.model(img_tensor)

                # Process predictions
                boxes = prediction[0]['boxes'].cpu().numpy()
                scores = prediction[0]['scores'].cpu().numpy()
                labels = prediction[0]['labels'].cpu().numpy()

                # Filter vehicle classes and confidence threshold
                vehicle_detections = []
                for box, score, label in zip(boxes, scores, labels):
                    if score >= self.confidence_threshold and label in self.vehicle_classes:
                        x1, y1, x2, y2 = box
                        vehicle_detections.append([[int(x1), int(y1), int(x2), int(y2)], score, int(label)])

                return vehicle_detections

            elif self.model_type == "yolov8":
                # YOLOv8 detection
                results = self.model(image, verbose=False)

                # Process results
                vehicle_detections = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        cls = int(box.cls)

                        # Filter for vehicles and confidence threshold
                        # YOLOv8 classes: 2=car, 5=bus, 7=truck
                        if conf >= self.confidence_threshold and cls in [2, 5, 7]:
                            vehicle_detections.append([[int(x1), int(y1), int(x2), int(y2)], conf, cls])

                return vehicle_detections

            elif self.model_type == "opencv":
                # OpenCV DNN detection
                height, width = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
                self.model.setInput(blob)
                detections = self.model.forward()

                vehicle_detections = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    class_id = int(detections[0, 0, i, 1])

                    # Filter for vehicles and confidence threshold
                    if confidence >= self.confidence_threshold and class_id in self.vehicle_classes:
                        # Extract bounding box
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        x1, y1, x2, y2 = box.astype("int")
                        vehicle_detections.append([[x1, y1, x2, y2], confidence, class_id])

                return vehicle_detections

            return []

        except Exception as e:
            print(f"Error in detect_vehicles: {e}")
            return []

    def set_confidence_threshold(self, threshold):
        """Update the confidence threshold"""
        self.confidence_threshold = threshold