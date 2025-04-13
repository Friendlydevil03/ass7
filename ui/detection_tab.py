from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from datetime import datetime
from utils.video_utils import list_available_videos
from utils.image_processor import process_parking_spaces, detect_vehicles_traditional, process_ml_detections
from utils.tracker_integration import initialize_tracker, process_ml_detections_with_tracking


class DetectionTab:
    """
    The Detection Tab handles video processing and display
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # Setup main frames
        self.main_frame = Frame(parent)
        self.main_frame.grid(row=0, column=0, sticky=NSEW)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=0)

        # Video display frame
        self.video_frame = Frame(self.main_frame, bg="black")
        self.video_frame.grid(row=0, column=0, sticky=NSEW, padx=5, pady=5)
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_canvas = Canvas(self.video_frame, bg="black")
        self.video_canvas.pack(fill=BOTH, expand=True)

        # Settings panel frame
        self.settings_frame = ttk.Frame(self.main_frame)
        self.settings_frame.grid(row=0, column=1, sticky=NSEW, padx=5, pady=5)

        # Add scrollbar to settings panel
        settings_canvas = Canvas(self.settings_frame)
        settings_scrollbar = ttk.Scrollbar(self.settings_frame, orient="vertical", command=settings_canvas.yview)
        settings_scrollable_frame = ttk.Frame(settings_canvas)

        settings_scrollable_frame.bind(
            "<Configure>",
            lambda e: settings_canvas.configure(
                scrollregion=settings_canvas.bbox("all")
            )
        )

        settings_canvas.create_window((0, 0), window=settings_scrollable_frame, anchor="nw")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)

        settings_canvas.pack(side="left", fill="both", expand=True)
        settings_scrollbar.pack(side="right", fill="y")

        # Replace settings_frame with the scrollable frame for all widget placement
        self.settings_frame = settings_scrollable_frame

        # Detection Mode selector
        mode_frame = ttk.LabelFrame(self.settings_frame, text="Detection Mode")
        mode_frame.pack(fill=X, padx=10, pady=5, expand=False)

        # Mode selection radio buttons
        # Simultaneous detection option
        self.simultaneous_frame = ttk.LabelFrame(self.settings_frame, text="Simultaneous Detection")
        self.simultaneous_frame.pack(fill=X, padx=10, pady=5, expand=False)

        self.simultaneous_var = BooleanVar(value=False)
        ttk.Checkbutton(self.simultaneous_frame, text="Run parking and vehicle detection simultaneously",
                        variable=self.simultaneous_var, command=self.toggle_simultaneous_mode).pack(
            anchor=W, padx=10, pady=2)
        self.mode_var = StringVar(value=self.app.detection_mode)
        ttk.Radiobutton(mode_frame, text="Parking Detection",
                        variable=self.mode_var, value="parking",
                        command=self.on_mode_change).pack(anchor=W, padx=10, pady=2)
        ttk.Radiobutton(mode_frame, text="Vehicle Counting",
                        variable=self.mode_var, value="vehicle",
                        command=self.on_mode_change).pack(anchor=W, padx=10, pady=2)

        # Video source selection
        self.video_source_frame = ttk.LabelFrame(self.settings_frame, text="Video Source")
        self.video_source_frame.pack(fill=X, padx=10, pady=5, expand=False)

        # Video source dropdown
        video_source_label = ttk.Label(self.video_source_frame, text="Source:")
        video_source_label.pack(side=LEFT, padx=5)

        self.video_source_var = StringVar()
        self.video_source_dropdown = ttk.Combobox(self.video_source_frame,
                                                  textvariable=self.video_source_var,
                                                  state="readonly", width=20)
        self.video_source_dropdown["values"] = self.app.video_sources
        if len(self.app.video_sources) > 0:
            self.video_source_dropdown.current(0)
        self.video_source_dropdown.pack(side=LEFT, padx=5, pady=5)
        self.video_source_dropdown.bind("<<ComboboxSelected>>", self.on_video_source_change)

        # Browse button for additional videos
        ttk.Button(self.video_source_frame, text="Browse...",
                   command=self.browse_video).pack(side=LEFT, padx=5)

        # Start/Stop detection
        self.detection_button_frame = ttk.Frame(self.settings_frame)
        self.detection_button_frame.pack(fill=X, padx=10, pady=5)

        self.detection_button_var = StringVar(value="Start Detection")
        self.detection_button = ttk.Button(self.detection_button_frame,
                                           textvariable=self.detection_button_var,
                                           command=self.toggle_detection)
        self.detection_button.pack(fill=X, pady=5)

        # Parking detection settings
        self.parking_settings_frame = ttk.LabelFrame(self.settings_frame,
                                                     text="Parking Detection Settings")
        self.parking_settings_frame.pack(fill=X, padx=10, pady=5, expand=False)

        # Threshold slider
        threshold_frame = ttk.Frame(self.parking_settings_frame)
        threshold_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(threshold_frame, text="Threshold:").pack(side=LEFT)

        self.threshold_var = IntVar(value=self.app.parking_threshold)
        threshold_slider = ttk.Scale(threshold_frame, from_=100, to=1000,
                                     orient=HORIZONTAL, variable=self.threshold_var)
        threshold_slider.pack(side=LEFT, fill=X, expand=True, padx=5)
        threshold_slider.bind("<ButtonRelease-1>", self.update_threshold)

        # Create StringVar for formatted display
        self.threshold_str_var = StringVar(value=str(self.app.parking_threshold))
        threshold_label = ttk.Label(threshold_frame, textvariable=self.threshold_str_var)
        threshold_label.pack(side=LEFT, padx=5)

        # Set up trace for live updates while dragging
        self.threshold_var.trace_add("write", self.update_threshold_display)

        # Debug mode
        debug_frame = ttk.Frame(self.parking_settings_frame)
        debug_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(debug_frame, text="Debug Mode:").pack(side=LEFT)
        self.debug_var = StringVar(value="Off")
        ttk.Radiobutton(debug_frame, text="On", variable=self.debug_var, value="On").pack(side=LEFT)
        ttk.Radiobutton(debug_frame, text="Off", variable=self.debug_var, value="Off").pack(side=LEFT)

        # Vehicle detection settings
        self.vehicle_settings_frame = ttk.LabelFrame(self.settings_frame,
                                                     text="Vehicle Detection Settings")

        # Line Height slider
        line_frame = ttk.Frame(self.vehicle_settings_frame)
        line_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(line_frame, text="Line Height:").pack(side=LEFT)
        self.line_var = IntVar(value=self.app.line_height)
        line_slider = ttk.Scale(line_frame, from_=50, to=700,
                                orient=HORIZONTAL, variable=self.line_var)
        line_slider.pack(side=LEFT, fill=X, expand=True, padx=5)
        line_slider.bind("<ButtonRelease-1>", self.update_line_height)

        # Create StringVar for formatted display
        self.line_str_var = StringVar(value=str(self.app.line_height))
        line_label = ttk.Label(line_frame, textvariable=self.line_str_var)
        line_label.pack(side=LEFT, padx=5)

        # Set up trace for live updates while dragging
        self.line_var.trace_add("write", self.update_line_display)

        # Min contour size slider
        contour_frame = ttk.Frame(self.vehicle_settings_frame)
        contour_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(contour_frame, text="Min Size:").pack(side=LEFT)
        self.contour_var = IntVar(value=self.app.min_contour_width)
        contour_slider = ttk.Scale(contour_frame, from_=10, to=100,
                                   orient=HORIZONTAL, variable=self.contour_var)
        contour_slider.pack(side=LEFT, fill=X, expand=True, padx=5)
        contour_slider.bind("<ButtonRelease-1>", self.update_contour_size)

        # Create StringVar for formatted display
        self.contour_str_var = StringVar(value=str(self.app.min_contour_width))
        contour_label = ttk.Label(contour_frame, textvariable=self.contour_str_var)
        contour_label.pack(side=LEFT, padx=5)

        # Set up trace for live updates while dragging
        self.contour_var.trace_add("write", self.update_contour_display)

        # Offset slider
        offset_frame = ttk.Frame(self.vehicle_settings_frame)
        offset_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(offset_frame, text="Offset:").pack(side=LEFT)
        self.offset_var = IntVar(value=self.app.offset)
        offset_slider = ttk.Scale(offset_frame, from_=5, to=50,
                                  orient=HORIZONTAL, variable=self.offset_var)
        offset_slider.pack(side=LEFT, fill=X, expand=True, padx=5)
        offset_slider.bind("<ButtonRelease-1>", self.update_offset)

        # Create StringVar for formatted display
        self.offset_str_var = StringVar(value=str(self.app.offset))
        offset_label = ttk.Label(offset_frame, textvariable=self.offset_str_var)
        offset_label.pack(side=LEFT, padx=5)

        # Set up trace for live updates while dragging
        self.offset_var.trace_add("write", self.update_offset_display)

        # Reset counter button
        reset_frame = ttk.Frame(self.vehicle_settings_frame)
        reset_frame.pack(fill=X, padx=5, pady=5)

        ttk.Button(reset_frame, text="Reset Counter",
                   command=self.reset_counter).pack(fill=X)

        # ML detection settings
        self.ml_frame = ttk.LabelFrame(self.settings_frame, text="Detection Method")
        self.ml_frame.pack(fill=X, padx=10, pady=5, expand=False)

        # ML Method options
        ml_method_frame = ttk.Frame(self.ml_frame)
        ml_method_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(ml_method_frame, text="Method:").pack(side=LEFT)
        self.ml_method_var = StringVar(value="Faster R-CNN")
        ml_method_options = ["Faster R-CNN", "YOLO + DeepSORT"]
        ml_method_dropdown = ttk.Combobox(ml_method_frame, textvariable=self.ml_method_var,
                                          values=ml_method_options, state="readonly", width=15)
        ml_method_dropdown.pack(side=LEFT, padx=5)
        ml_method_dropdown.bind("<<ComboboxSelected>>", self.on_ml_method_change)

        # Use ML Detection checkbox
        ml_checkbox_frame = ttk.Frame(self.ml_frame)
        ml_checkbox_frame.pack(fill=X, padx=5, pady=5)

        self.ml_var = BooleanVar(value=False)
        self.ml_checkbox = ttk.Checkbutton(ml_checkbox_frame, text="Use ML Detection",
                                           variable=self.ml_var, command=self.on_ml_toggle)
        self.ml_checkbox.pack(side=LEFT)

        # ML Confidence setting
        confidence_frame = ttk.Frame(self.ml_frame)
        confidence_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(confidence_frame, text="Confidence:").pack(side=LEFT)
        self.confidence_var = DoubleVar(value=self.app.ml_confidence)
        confidence_slider = ttk.Scale(confidence_frame, from_=0.1, to=0.9,
                                      variable=self.confidence_var, orient=HORIZONTAL)
        confidence_slider.pack(side=LEFT, fill=X, expand=True, padx=5)
        confidence_slider.bind("<ButtonRelease-1>", self.on_confidence_change)

        # Create StringVar for formatted display
        self.confidence_str_var = StringVar(value=f"{self.app.ml_confidence:.2f}")
        confidence_label = ttk.Label(confidence_frame, textvariable=self.confidence_str_var)
        confidence_label.pack(side=LEFT, padx=5)

        # Set up trace for live updates while dragging
        self.confidence_var.trace_add("write", self.update_confidence_display)

        # Tracking options frame (initially hidden)
        self.tracking_frame = ttk.LabelFrame(self.settings_frame, text="Tracking Settings")

        # Trail display option
        trail_frame = ttk.Frame(self.tracking_frame)
        trail_frame.pack(fill=X, padx=5, pady=5)

        self.show_trail_var = BooleanVar(value=True)
        trail_checkbox = ttk.Checkbutton(trail_frame, text="Show Tracking Trails",
                                         variable=self.show_trail_var)
        trail_checkbox.pack(side=LEFT)

        # Status display
        status_frame = ttk.LabelFrame(self.settings_frame, text="Status")
        status_frame.pack(fill=X, padx=10, pady=5, expand=False)

        # Status labels
        self.spaces_label = ttk.Label(status_frame, text="Spaces: 0 / 0")
        self.spaces_label.pack(anchor=W, padx=5, pady=2)

        self.free_label = ttk.Label(status_frame, text="Free Spaces: 0")
        self.free_label.pack(anchor=W, padx=5, pady=2)

        self.occupied_label = ttk.Label(status_frame, text="Occupied Spaces: 0")
        self.occupied_label.pack(anchor=W, padx=5, pady=2)

        self.vehicles_label = ttk.Label(status_frame, text="Vehicles Counted: 0")
        self.vehicles_label.pack(anchor=W, padx=5, pady=2)

        # ML status label
        self.ml_status_label = ttk.Label(status_frame, text="ML Detection: Disabled",
                                         foreground="grey")
        self.ml_status_label.pack(anchor=W, padx=5, pady=2)

        # Processed time tracking label
        self.processing_time_label = ttk.Label(status_frame, text="Processing: 0 ms")
        self.processing_time_label.pack(anchor=W, padx=5, pady=2)

  
        # Initialize video settings
        self.running = False
        self.video_capture = None
        self.prev_frame = None
        self.frame_count = 0
        self.frame_skip = 2
        self.last_processing_time = 0

        # Show appropriate settings based on mode
        self.on_mode_change()

    # Methods for formatted slider displays
    def update_threshold_display(self, *args):
        """Format the threshold value as an integer"""
        try:
            value = self.threshold_var.get()
            self.threshold_str_var.set(f"{int(value)}")
        except:
            pass

    def update_line_display(self, *args):
        """Format the line height value as an integer"""
        try:
            value = self.line_var.get()
            self.line_str_var.set(f"{int(value)}")
        except:
            pass

    def update_contour_display(self, *args):
        """Format the contour size value as an integer"""
        try:
            value = self.contour_var.get()
            self.contour_str_var.set(f"{int(value)}")
        except:
            pass

    def update_offset_display(self, *args):
        """Format the offset value as an integer"""
        try:
            value = self.offset_var.get()
            self.offset_str_var.set(f"{int(value)}")
        except:
            pass

    def update_confidence_display(self, *args):
        """Format the confidence value to 2 decimal places"""
        try:
            value = self.confidence_var.get()
            self.confidence_str_var.set(f"{value:.2f}")
        except:
            pass

    def on_ml_method_change(self, event=None):
        """Handle ML method change"""
        method = self.ml_method_var.get()

        # Show/hide tracking settings based on method
        if method == "YOLO + DeepSORT":
            self.tracking_frame.pack(fill=X, padx=10, pady=5, expand=False, after=self.ml_frame)
            self.app.use_yolo_tracking = True
        else:
            self.tracking_frame.pack_forget()
            self.app.use_yolo_tracking = False

        # If ML detection is currently enabled, reinitialize
        if self.app.use_ml_detection:
            self.on_ml_toggle()
            self.on_ml_toggle()

    def on_mode_change(self, event=None):
        """Handle detection mode change"""
        mode = self.mode_var.get()
        self.app.detection_mode = mode

        # Show/hide appropriate setting frames
        if mode == "parking":
            self.parking_settings_frame.pack(after=self.detection_button_frame,
                                             fill=X, padx=10, pady=5, expand=False)
            self.vehicle_settings_frame.pack_forget()
        else:  # vehicle mode
            self.vehicle_settings_frame.pack(after=self.detection_button_frame,
                                             fill=X, padx=10, pady=5, expand=False)
            self.parking_settings_frame.pack_forget()

    def on_video_source_change(self, event=None):
        """Handle video source change"""
        if self.running:
            self.stop_detection()
            self.toggle_detection()

    def browse_video(self):
        """Browse for a video file"""
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mov *.mkv'),
            ('All files', '*.*')
        )

        filepath = filedialog.askopenfilename(
            title='Select a video file',
            filetypes=filetypes
        )

        if filepath:
            # Add to dropdown if not already there
            if filepath not in self.video_source_dropdown["values"]:
                values = list(self.video_source_dropdown["values"])
                values.append(filepath)
                self.video_source_dropdown["values"] = values

            # Set as current selection
            self.video_source_var.set(filepath)

    def toggle_detection(self):
        """Start or stop detection"""
        if not self.running:
            # Update the reference image dimensions before starting
            if self.app.current_reference_image in self.app.reference_dimensions:
                ref_dimensions = self.app.reference_dimensions[self.app.current_reference_image]
                self.app.log_event(f"Using reference dimensions: {ref_dimensions}")

            # Now start detection with proper scaling
            self.start_detection()
        else:
            self.stop_detection()

    def toggle_simultaneous_mode(self):
        """Toggle simultaneous detection mode"""
        simultaneous_enabled = self.simultaneous_var.get()

        if simultaneous_enabled:
            # Create video source selection dialogs
            self.open_video_selection_dialog()
        else:
            # If we have running dialogs, close them
            if hasattr(self, 'parking_dialog') and self.parking_dialog:
                self.parking_dialog.close_dialog()
                self.parking_dialog = None

            if hasattr(self, 'vehicle_dialog') and self.vehicle_dialog:
                self.vehicle_dialog.close_dialog()
                self.vehicle_dialog = None

            # Re-enable standard mode selection
            mode_frame = None
            for child in self.settings_frame.winfo_children():
                if isinstance(child, ttk.LabelFrame) and child.winfo_children() and \
                        len(child.winfo_children()) > 0 and "Detection Mode" in child.cget("text"):
                    mode_frame = child
                    break

            if mode_frame:
                for child in mode_frame.winfo_children():
                    if isinstance(child, ttk.Radiobutton):
                        child.configure(state="normal")
                self.mode_var.set(self.app.detection_mode)

    def open_video_selection_dialog(self):
        """Open a dialog to select video sources for parking and vehicle detection"""
        selection_dialog = Toplevel(self.parent)
        selection_dialog.title("Select Video Sources")
        selection_dialog.geometry("400x300")
        selection_dialog.grab_set()  # Make dialog modal

        # Create frames
        main_frame = ttk.Frame(selection_dialog, padding=10)
        main_frame.pack(fill=BOTH, expand=True)

        # Parking detection video source
        parking_frame = ttk.LabelFrame(main_frame, text="Parking Detection Video Source")
        parking_frame.pack(fill=X, pady=10)

        parking_source_var = StringVar()
        parking_combobox = ttk.Combobox(parking_frame, textvariable=parking_source_var, width=30)
        parking_combobox["values"] = self.video_source_dropdown["values"]
        if len(parking_combobox["values"]) > 0:
            parking_combobox.current(0)
        parking_combobox.pack(pady=5, padx=5)

        ttk.Button(parking_frame, text="Browse...",
                   command=lambda: self.browse_video_for_dialog(parking_combobox)).pack(pady=5)

        # Vehicle detection video source
        vehicle_frame = ttk.LabelFrame(main_frame, text="Vehicle Detection Video Source")
        vehicle_frame.pack(fill=X, pady=10)

        vehicle_source_var = StringVar()
        vehicle_combobox = ttk.Combobox(vehicle_frame, textvariable=vehicle_source_var, width=30)
        vehicle_combobox["values"] = self.video_source_dropdown["values"]
        if len(vehicle_combobox["values"]) > 1:
            vehicle_combobox.current(1)
        elif len(vehicle_combobox["values"]) > 0:
            vehicle_combobox.current(0)
        vehicle_combobox.pack(pady=5, padx=5)

        ttk.Button(vehicle_frame, text="Browse...",
                   command=lambda: self.browse_video_for_dialog(vehicle_combobox)).pack(pady=5)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=10)

        ttk.Button(button_frame, text="Cancel",
                   command=lambda: self.cancel_video_selection(selection_dialog)).pack(side=RIGHT, padx=5)

        ttk.Button(button_frame, text="Start Detection",
                   command=lambda: self.start_simultaneous_detection(
                       parking_source_var.get(),
                       vehicle_source_var.get(),
                       selection_dialog
                   )).pack(side=RIGHT, padx=5)

    def browse_video_for_dialog(self, combobox):
        """Browse for a video file and update the combobox"""
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mov *.mkv'),
            ('All files', '*.*')
        )

        filepath = filedialog.askopenfilename(
            title='Select a video file',
            filetypes=filetypes
        )

        if filepath:
            # Add to dropdown if not already there
            current_values = list(combobox["values"])
            if filepath not in current_values:
                current_values.append(filepath)
                combobox["values"] = current_values

            # Set as current selection
            combobox.set(filepath)

    def cancel_video_selection(self, dialog):
        """Cancel the video selection dialog"""
        self.simultaneous_var.set(False)
        dialog.destroy()

    def start_simultaneous_detection(self, parking_source, vehicle_source, dialog):
        """Start the simultaneous detection in separate windows"""
        from ui.detection_dialog import DetectionDialog

        try:
            # Make sure we have video sources
            if not parking_source or not vehicle_source:
                messagebox.showerror("Error", "Please select video sources for both detection types")
                return

            # Create the detection dialogs
            self.parking_dialog = DetectionDialog(self.parent, self.app, "parking", parking_source)
            self.vehicle_dialog = DetectionDialog(self.parent, self.app, "vehicle", vehicle_source)

            # Close the selection dialog
            dialog.destroy()

            # Find the mode_frame and disable mode selection
            mode_frame = None
            for child in self.settings_frame.winfo_children():
                if isinstance(child, ttk.LabelFrame) and child.winfo_children() and \
                        len(child.winfo_children()) > 0 and "Detection Mode" in child.cget("text"):
                    mode_frame = child
                    break

            if mode_frame:
                for child in mode_frame.winfo_children():
                    if isinstance(child, ttk.Radiobutton):
                        child.configure(state="disabled")

            # Stop any running detection in the main window
            if self.running:
                self.stop_detection()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simultaneous detection: {str(e)}")
            self.simultaneous_var.set(False)
            dialog.destroy()

    def start_detection(self):
        """Start video detection"""
        try:
            # Get selected video source
            video_source = self.video_source_var.get()

            # Convert 'Webcam' to integer index
            if video_source == "Webcam":
                video_source = 0

            # Open video capture
            self.video_capture = cv2.VideoCapture(video_source)

            # Check if opened successfully
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", f"Failed to open video source: {video_source}")
                return

            # Update UI
            self.running = True
            self.detection_button_var.set("Stop Detection")

            # Start frame processing
            self.process_frame()

            # Update app current video
            self.app.current_video = video_source

            # If this is a webcam and there's a reference image mapping:
            if video_source == 0 and "0" in self.app.video_reference_map:
                ref_image = self.app.video_reference_map["0"]
                if ref_image != self.app.current_reference_image:
                    self.app.current_reference_image = ref_image
                    self.app.load_parking_positions(ref_image)

            # For other videos, check reference mapping
            elif isinstance(video_source, str) and video_source in self.app.video_reference_map:
                ref_image = self.app.video_reference_map[video_source]
                if ref_image != self.app.current_reference_image:
                    self.app.current_reference_image = ref_image
                    self.app.load_parking_positions(ref_image)

        except Exception as e:
            self.app.log_event(f"Error starting detection: {str(e)}")
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")

    def stop_detection(self):
        """Stop video detection"""
        self.running = False
        self.detection_button_var.set("Start Detection")

        # Release video capture
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        # Clear previous frame
        self.prev_frame = None

        # Reset frame count
        self.frame_count = 0

    def update_threshold(self, event=None):
        """Update parking threshold value"""
        self.app.parking_threshold = self.threshold_var.get()

    def update_line_height(self, event=None):
        """Update line height value"""
        self.app.line_height = self.line_var.get()

    def update_contour_size(self, event=None):
        """Update minimum contour size"""
        size = self.contour_var.get()
        self.app.min_contour_width = size
        self.app.min_contour_height = size

    def update_offset(self, event=None):
        """Update offset value"""
        self.app.offset = self.offset_var.get()

    def reset_counter(self):
        """Reset vehicle counter"""
        self.app.vehicle_counter = 0
        self.app.matches = []
        if hasattr(self.app, 'vehicle_tracker') and self.app.vehicle_tracker:
            self.app.vehicle_tracker.reset_count()
        self.update_status_info(
            self.app.total_spaces,
            self.app.free_spaces,
            self.app.occupied_spaces,
            self.app.vehicle_counter
        )

    def on_ml_toggle(self):
        """Toggle ML detection on/off"""
        ml_enabled = self.ml_var.get()
        self.app.use_ml_detection = ml_enabled

        # Update confidence setting in app
        self.app.ml_confidence = self.confidence_var.get()

        if ml_enabled:
            try:
                # Check which ML method to use
                ml_method = self.ml_method_var.get()

                if ml_method == "YOLO + DeepSORT":
                    # Initialize the YOLO + DeepSORT tracker
                    self.app.ml_detector = initialize_tracker(
                        confidence_threshold=self.app.ml_confidence,
                        use_cuda=True  # You can make this configurable
                    )

                    # Store in a separate variable for tracking
                    self.app.vehicle_tracker = self.app.ml_detector
                    self.app.log_event("YOLO + DeepSORT tracker initialized")

                else:
                    # Initialize the original ML detector
                    self.app.log_event("Initializing ML detector...")
                    from models.vehicle_detector import VehicleDetector
                    self.app.ml_detector = VehicleDetector(confidence_threshold=self.app.ml_confidence)
                    self.app.vehicle_tracker = None
                    self.app.log_event("ML detector initialized")

                # Show success message
                self.ml_status_label.config(text="ML Detection: Active", foreground="green")

            except Exception as e:
                self.ml_var.set(False)
                self.app.use_ml_detection = False
                error_msg = f"Failed to initialize ML detection: {str(e)}"
                self.app.log_event(error_msg)
                self.ml_status_label.config(text="ML Detection: Error", foreground="red")
                messagebox.showerror("ML Initialization Error", error_msg)
        else:
            # Disable ML detection
            self.app.ml_detector = None
            self.app.vehicle_tracker = None
            self.ml_status_label.config(text="ML Detection: Disabled", foreground="grey")

    def on_confidence_change(self, event=None):
        """Update ML confidence threshold"""
        self.app.ml_confidence = self.confidence_var.get()
        if self.app.use_ml_detection and self.app.ml_detector:
            self.app.ml_detector.set_confidence_threshold(self.app.ml_confidence)

    def update_status_info(self, total_spaces, free_spaces, occupied_spaces, vehicle_count):
        """Update status information displays"""
        self.spaces_label.config(text=f"Spaces: {total_spaces}")
        self.free_label.config(text=f"Free Spaces: {free_spaces}")
        self.occupied_label.config(text=f"Occupied Spaces: {occupied_spaces}")
        self.vehicles_label.config(text=f"Vehicles Counted: {vehicle_count}")

        # Update process_frame method (around line 573)

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
                if isinstance(self.video_source_var.get(), str) and not self.video_source_var.get() == "Webcam":
                    self.app.log_event("End of video reached")
                    self.stop_detection()
                else:
                    # For webcam, this could be a temporary error
                    self.parent.after(100, self.process_frame)
                return

            # Resize frame for display if needed
            original_height, original_width = img.shape[:2]
            if original_width != self.app.image_width or original_height != self.app.image_height:
                self.app.image_width = original_width
                self.app.image_height = original_height

                # Scale parking positions if needed (only for parking detection)
                if self.app.detection_mode == "parking":
                    self.app.scale_positions_to_current_dimensions()

            # Process the frame based on detection mode
            processed_img = None

            # Ensure dimensions are correctly updated before scaling
            if self.app.current_reference_image in self.app.reference_dimensions:
                ref_width, ref_height = self.app.reference_dimensions[self.app.current_reference_image]
                if ref_width != original_width or ref_height != original_height:
                    self.app.log_event(
                        f"Updating dimensions from {ref_width}x{ref_height} to {original_width}x{original_height}")

            if self.app.detection_mode == "parking":
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

                # Get image dimensions to compute scale factors
                img_height, img_width = img.shape[:2]

                # Calculate scale factors if using small processing image
                processing_img = img.copy()
                width_scale = 1.0
                height_scale = 1.0

                # Process with scaled positions and threshold
                debug_mode = hasattr(self, 'debug_var') and self.debug_var.get() == "On"
                processed_small_img, free_spaces, occupied_spaces, total_spaces = process_parking_spaces(
                    imgProcessed, processing_img.copy(), scaled_positions,
                    int(self.app.parking_threshold * width_scale), debug=debug_mode
                )

                # Scale back up for display if needed
                processed_img = cv2.resize(processed_small_img, (self.app.image_width, self.app.image_height))

                # Update app state
                self.app.free_spaces = free_spaces
                self.app.occupied_spaces = occupied_spaces
                self.app.total_spaces = total_spaces

                # Update allocation data
                self.update_parking_data_for_allocation(imgProcessed)

            elif self.app.detection_mode == "vehicle":
                # Initialize the frame if needed
                if self.prev_frame is None or self.frame_count == 0:
                    self.prev_frame = img.copy()
                    self.frame_count = 1

                    # Schedule next frame and return
                    self.parent.after(30, self.process_frame)
                    return

                self.frame_count += 1

                # Adjust frame skip rate for vehicle detection
                self.frame_skip = 8 if self.app.use_ml_detection else 4

                # Check if we should use ML detection
                if self.app.use_ml_detection and self.app.ml_detector:
                    try:
                        # Check if we're using YOLO + DeepSORT
                        ml_method = getattr(self, 'ml_method_var', None)
                        if ml_method and ml_method.get() == "YOLO + DeepSORT" and hasattr(self.app, 'vehicle_tracker'):
                            # Process with tracking
                            processed_img, new_matches, new_vehicle_counter = process_ml_detections_with_tracking(
                                img.copy(),
                                self.app.vehicle_tracker,
                                self.app.line_height,
                                self.app.offset,
                                self.app.vehicle_counter,
                                self.app.ml_detector.classes if hasattr(self.app.ml_detector, 'classes') else []
                            )

                            # Update app state
                            self.app.matches = new_matches
                            self.app.vehicle_counter = new_vehicle_counter

                            # Update the processed image
                            img = processed_img
                        else:
                            # Only run ML detection on certain frames to improve performance
                            if self.frame_count % self.frame_skip == 0:
                                # Use our safe detection method
                                detections = self.safe_ml_detection(img)

                                # Store for use in skipped frames
                                self.last_detections = detections
                            else:
                                # Use the last known detections for in-between frames
                                detections = self.last_detections if hasattr(self,
                                                                             'last_detections') and self.last_detections is not None else []

                            # Check if we have valid detections to process
                            if not isinstance(detections, list):
                                raise TypeError(f"Expected list of detections but got {type(detections)}")

                            # Process the ML detections
                            processed_img, new_matches, new_vehicle_counter = process_ml_detections(
                                img.copy(),
                                detections,
                                self.app.line_height,
                                self.app.offset,
                                self.app.matches,
                                self.app.vehicle_counter,
                                self.app.ml_detector.classes if hasattr(self.app.ml_detector, 'classes') else []
                            )

                            # Update app state
                            self.app.matches = new_matches
                            self.app.vehicle_counter = new_vehicle_counter

                            # Update the processed image
                            img = processed_img

                    except Exception as e:
                        print(f"ML detection error: {str(e)}")
                        self.app.log_event(f"ML detection error: {str(e)}")

                        # Fallback to traditional method
                        processed_img, new_matches, new_vehicle_counter = detect_vehicles_traditional(
                            img.copy(),
                            self.prev_frame,
                            self.app.line_height,
                            self.app.min_contour_width,
                            self.app.min_contour_height,
                            self.app.offset,
                            self.app.matches,
                            self.app.vehicle_counter
                        )
                else:
                    # Use traditional vehicle detection
                    processed_img, new_matches, new_vehicle_counter = detect_vehicles_traditional(
                        img.copy(),
                        self.prev_frame,
                        self.app.line_height,
                        self.app.min_contour_width,
                        self.app.min_contour_height,
                        self.app.offset,
                        self.app.matches,
                        self.app.vehicle_counter
                    )

                # Update app state
                self.app.matches = new_matches
                self.app.vehicle_counter = new_vehicle_counter

            # Use the original image if no processing was done
            if processed_img is None:
                processed_img = img.copy()

            # Update the previous frame for the next iteration
            self.prev_frame = img.copy()

            # Convert to RGB for display
            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

            # Convert to PIL format
            img_pil = Image.fromarray(img_rgb)

            # Reuse ImageTk object if possible
            if not hasattr(self, 'img_tk') or self.img_tk is None:
                self.img_tk = ImageTk.PhotoImage(image=img_pil)
            else:
                # Create a new PhotoImage (can't update existing one)
                self.img_tk = ImageTk.PhotoImage(image=img_pil)

            # Display the image
            if hasattr(self, 'image_label'):
                self.image_label.configure(image=self.img_tk)
                self.image_label.image = self.img_tk
            else:
                self.image_label = Label(self.video_canvas, image=self.img_tk)
                self.image_label.pack(fill=BOTH, expand=True)
                self.image_label.image = self.img_tk

            # Update status information
            self.update_status_info(
                self.app.total_spaces,
                self.app.free_spaces,
                self.app.occupied_spaces,
                self.app.vehicle_counter
            )

            # Update allocation tab less frequently (every ~10 seconds at 30fps)
            if hasattr(self.app, 'allocation_tab'):
                # Assuming ~30fps, update every 300 frames (10 seconds)
                if self.frame_count % 300 == 0:
                    self.app.allocation_tab.update_visualization()
                    self.app.allocation_tab.update_statistics()

            # Calculate and display processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.last_processing_time = processing_time
            self.processing_time_label.config(text=f"Processing: {processing_time:.1f} ms")

            # Schedule next frame processing with better delay
            if self.app.detection_mode == "parking" or not self.app.use_ml_detection:
                self.parent.after(30, self.process_frame)  # 30ms delay for standard processing
            else:
                self.parent.after(40, self.process_frame)  # 40ms delay for ML processing

        except Exception as e:
            self.app.log_event(f"Error processing frame: {str(e)}")
            messagebox.showerror("Error", f"Error processing video frame: {str(e)}")
            self.stop_detection()

    def update_parking_data_for_allocation(self, img_pro):
        """Update parking data for allocation system"""
        try:
            # Make sure app has parking_manager
            if not hasattr(self.app, 'parking_manager'):
                self.app.log_event("No parking manager found")
                return

            # Create parking_data if it doesn't exist in parking_manager
            if not hasattr(self.app.parking_manager, 'parking_data'):
                self.app.parking_manager.parking_data = {}

            # Update parking spaces data
            for i, (x, y, w, h) in enumerate(self.app.posList):
                # Convert coordinates to integers to fix the "slice indices must be integers" error
                x, y, w, h = int(x), int(y), int(w), int(h)

                # Ensure coordinates are within image bounds
                if (y >= 0 and y + h < img_pro.shape[0] and x >= 0 and x + w < img_pro.shape[1]):
                    # Get crop of parking space
                    img_crop = img_pro[y:y + h, x:x + w]
                    count = cv2.countNonZero(img_crop)
                    is_occupied = count >= self.app.parking_threshold

                    # Generate section based on position (cast to int to avoid float division issues)
                    section = "A" if x < int(img_pro.shape[1] / 2) else "B"
                    section += "1" if y < int(img_pro.shape[0] / 2) else "2"

                    # Full space ID
                    space_id = f"S{i + 1}-{section}"

                    # Update or create parking space data
                    if space_id not in self.app.parking_manager.parking_data:
                        self.app.parking_manager.parking_data[space_id] = {
                            'position': (x, y, w, h),
                            'occupied': is_occupied,
                            'vehicle_id': None,
                            'last_state_change': datetime.now(),
                            'distance_to_entrance': x + y,  # Simple distance estimation
                            'section': section
                        }
                    else:
                        # Just update occupancy status
                        self.app.parking_manager.parking_data[space_id]['occupied'] = is_occupied

            # Only log updates occasionally to reduce console spam
            if self.frame_count % 100 == 0:  # Log every 100 frames
                self.app.log_event(f"Updated parking data for {len(self.app.posList)} spaces")
        except Exception as e:
            self.app.log_event(f"Error updating parking allocation data: {str(e)}")

    def safe_ml_detection(self, img):
        """Safely perform ML detection with error handling and fallback"""
        try:
            if not self.app.ml_detector:
                return []

            # Check if we're using the tracker or regular detector
            ml_method = getattr(self, 'ml_method_var', None)
            if ml_method and ml_method.get() == "YOLO + DeepSORT" and hasattr(self.app, 'vehicle_tracker'):
                # For YOLO+DeepSORT, we don't need to do anything here
                # The detections will be handled in process_ml_detections_with_tracking
                return []

            # Use the regular detector
            # Create a smaller image for detection
            ml_img = cv2.resize(img, (640, 360))

            # Get vehicle detections
            detections = self.app.ml_detector.detect_vehicles(ml_img)

            # Ensure we have a valid result
            if detections is None:
                return []

            # Scale detection coordinates back to original image size
            if len(detections) > 0:
                width_scale = self.app.image_width / 640
                height_scale = self.app.image_height / 360

                for i, detection in enumerate(detections):
                    if len(detection) >= 3:
                        box = detection[0]
                        if len(box) >= 4:
                            x1, y1, x2, y2 = box
                            detections[i][0] = [
                                int(x1 * width_scale),
                                int(y1 * height_scale),
                                int(x2 * width_scale),
                                int(y2 * height_scale)
                            ]

            return detections

        except Exception as e:
            print(f"Error in ML detection: {str(e)}")
            return []