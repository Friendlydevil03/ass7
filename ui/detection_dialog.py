from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from datetime import datetime
from utils.image_processor import process_parking_spaces, detect_vehicles_traditional, process_ml_detections
from utils.tracker_integration import process_ml_detections_with_tracking


class DetectionDialog:
    """
    Dialog for running either parking or vehicle detection on a separate video source
    """

    def __init__(self, parent, app, detection_type, video_source):
        """
        Initialize a detection dialog window

        Args:
            parent: Parent window
            app: Main application reference
            detection_type: "parking" or "vehicle"
            video_source: Path to video or camera index
        """
        self.parent = parent
        self.app = app
        self.detection_type = detection_type
        self.video_source = video_source

        # Create dialog window
        self.dialog = Toplevel(parent)
        self.dialog.title(f"{detection_type.title()} Detection")
        self.dialog.geometry("800x600")
        self.dialog.protocol("WM_DELETE_WINDOW", self.close_dialog)

        # Setup main frames
        self.main_frame = Frame(self.dialog)
        self.main_frame.pack(fill=BOTH, expand=True)

        # Video display frame
        self.video_frame = Frame(self.main_frame, bg="black")
        self.video_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        self.video_canvas = Canvas(self.video_frame, bg="black")
        self.video_canvas.pack(fill=BOTH, expand=True)

        # Status frame
        self.status_frame = ttk.LabelFrame(self.main_frame, text="Status")
        self.status_frame.pack(fill=X, padx=5, pady=5)

        if detection_type == "parking":
            # Add status labels for parking
            self.spaces_label = ttk.Label(self.status_frame, text="Spaces: 0 / 0")
            self.spaces_label.pack(side=LEFT, padx=5, pady=2)

            self.free_label = ttk.Label(self.status_frame, text="Free: 0")
            self.free_label.pack(side=LEFT, padx=5, pady=2)

            self.occupied_label = ttk.Label(self.status_frame, text="Occupied: 0")
            self.occupied_label.pack(side=LEFT, padx=5, pady=2)
        else:
            # Add status labels for vehicle detection
            self.vehicles_label = ttk.Label(self.status_frame, text="Vehicles: 0")
            self.vehicles_label.pack(side=LEFT, padx=5, pady=2)

        # Processing time label
        self.processing_time_label = ttk.Label(self.status_frame, text="Processing: 0 ms")
        self.processing_time_label.pack(side=RIGHT, padx=5, pady=2)

        # Initialize video settings
        self.running = False
        self.video_capture = None
        self.prev_frame = None
        self.frame_count = 0
        self.frame_skip = 2
        self.last_processing_time = 0

        # Start the detection
        self.start_detection()

    def start_detection(self):
        """Start video detection"""
        try:
            video_source = self.video_source

            # Convert 'Webcam' to integer index
            if video_source == "Webcam":
                video_source = 0

            # Open video capture
            self.video_capture = cv2.VideoCapture(video_source)

            # Check if opened successfully
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", f"Failed to open video source: {video_source}")
                self.close_dialog()
                return

            # Update running state
            self.running = True

            # Start frame processing
            self.process_frame()

            # For parking detection, load positions if needed
            if self.detection_type == "parking":
                if isinstance(video_source, str) and video_source in self.app.video_reference_map:
                    ref_image = self.app.video_reference_map[video_source]
                    if ref_image != self.app.current_reference_image:
                        self.app.current_reference_image = ref_image
                        self.app.load_parking_positions(ref_image)

        except Exception as e:
            self.app.log_event(f"Error starting {self.detection_type} detection dialog: {str(e)}")
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
            self.close_dialog()

    def close_dialog(self):
        """Close the dialog and release resources"""
        self.running = False

        # Release video capture
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        # Destroy dialog
        self.dialog.destroy()

    def update_status_info(self, total_spaces=0, free_spaces=0, occupied_spaces=0, vehicle_count=0):
        """Update status information displays"""
        if self.detection_type == "parking":
            self.spaces_label.config(text=f"Spaces: {total_spaces}")
            self.free_label.config(text=f"Free: {free_spaces}")
            self.occupied_label.config(text=f"Occupied: {occupied_spaces}")
        else:
            self.vehicles_label.config(text=f"Vehicles: {vehicle_count}")

    def process_frame(self):
        """Process a video frame"""
        if not self.running or not self.video_capture:
            return

        try:
            start_time = time.time()

            # Read frame from video
            ret, img = self.video_capture.read()

            # Check if frame was read successfully
            if not ret:
                # For video files, this means end of video
                if isinstance(self.video_source, str) and not self.video_source == "Webcam":
                    self.app.log_event(f"End of video reached in {self.detection_type} dialog")
                    self.close_dialog()
                else:
                    # For webcam, this could be a temporary error
                    self.dialog.after(100, self.process_frame)
                return

            # Process the frame based on detection type
            processed_img = None

            if self.detection_type == "parking":
                # Convert to grayscale and blur for processing
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
                imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, 25, 16)
                imgProcessed = cv2.medianBlur(imgThreshold, 5)

                # Apply dilation and erosion to clean up
                kernel = np.ones((3, 3), np.uint8)
                imgProcessed = cv2.dilate(imgProcessed, kernel, iterations=1)
                imgProcessed = cv2.erode(imgProcessed, kernel, iterations=1)

                # Get scaled positions for current frame size
                scaled_positions = self.app.posList.copy()

                # Process with scaled positions and threshold
                debug_mode = False
                processed_small_img, free_spaces, occupied_spaces, total_spaces = process_parking_spaces(
                    imgProcessed, img.copy(), scaled_positions,
                    int(self.app.parking_threshold), debug=debug_mode
                )

                processed_img = processed_small_img

                # Update app state
                self.app.free_spaces = free_spaces
                self.app.occupied_spaces = occupied_spaces
                self.app.total_spaces = total_spaces

                # Update status display
                self.update_status_info(
                    total_spaces,
                    free_spaces,
                    occupied_spaces
                )

            elif self.detection_type == "vehicle":
                # Initialize the frame if needed
                if self.prev_frame is None or self.frame_count == 0:
                    self.prev_frame = img.copy()
                    self.frame_count = 1

                    # Schedule next frame and return
                    self.dialog.after(30, self.process_frame)
                    return

                self.frame_count += 1

                # Use traditional vehicle detection
                processed_img, new_matches, new_vehicle_counter = detect_vehicles_traditional(
                    img.copy(),
                    self.prev_frame,
                    self.app.line_height,
                    self.app.min_contour_width,
                    self.app.min_contour_height,
                    self.app.offset,
                    self.app.matches.copy() if hasattr(self.app, 'matches') else [],
                    self.app.vehicle_counter
                )

                # Update app state
                self.app.matches = new_matches
                self.app.vehicle_counter = new_vehicle_counter

                # Update status display
                self.update_status_info(vehicle_count=new_vehicle_counter)

                # Update the previous frame for the next iteration
                self.prev_frame = img.copy()

            # Use the original image if no processing was done
            if processed_img is None:
                processed_img = img.copy()

            # Convert to RGB for display
            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

            # Convert to PIL format
            img_pil = Image.fromarray(img_rgb)

            # Create a new PhotoImage
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Display the image
            if hasattr(self, 'image_label'):
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk
            else:
                self.image_label = Label(self.video_canvas, image=img_tk)
                self.image_label.pack(fill=BOTH, expand=True)
                self.image_label.image = img_tk

            # Calculate and display processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.last_processing_time = processing_time
            self.processing_time_label.config(text=f"Processing: {processing_time:.1f} ms")

            # Schedule next frame processing
            self.dialog.after(30, self.process_frame)

        except Exception as e:
            self.app.log_event(f"Error processing frame in {self.detection_type} dialog: {str(e)}")
            messagebox.showerror("Error", f"Error processing video frame: {str(e)}")
            self.close_dialog()