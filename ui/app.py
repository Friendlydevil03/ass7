import os
import threading
import time
from tkinter import Frame, Tk, messagebox, ttk
from tkinter import BOTH, TOP, BOTTOM, LEFT, RIGHT, X, Y, NSEW, W, E, N, S
from datetime import datetime
from PIL import Image, ImageTk
from models.allocation_engine import ParkingAllocationEngine
from ui.detection_tab import DetectionTab
from ui.setup_tab import SetupTab
from ui.log_tab import LogTab
from ui.stats_tab import StatsTab
from ui.reference_tab import ReferenceTab
from models.parking_visualizer import ParkingVisualizer
from models.allocation_engine import ParkingAllocationEngine
from ui.parking_allocation_tab import ParkingAllocationTab
from models.vehicle_detector import VehicleDetector
from utils.resource_manager import ensure_directories_exist, load_parking_positions
from utils.media_paths import list_available_videos

class ParkingManagementSystem:
    DEFAULT_CONFIDENCE = 0.6
    DEFAULT_THRESHOLD = 500
    MIN_CONTOUR_SIZE = 40
    DEFAULT_OFFSET = 10
    DEFAULT_LINE_HEIGHT = 400

    def __init__(self, master):
        self.master = master
        self.master.title("Smart Parking Management System")

        self.master.geometry("1280x720")
        self.master.minsize(800, 600)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize class variables
        self.running = False
        self.posList = []
        self.video_capture = None
        self.current_video = None
        self.vehicle_counter = 0
        self.matches = []  # For vehicle counting
        self.line_height = self.DEFAULT_LINE_HEIGHT
        self.min_contour_width = self.MIN_CONTOUR_SIZE
        self.min_contour_height = self.MIN_CONTOUR_SIZE
        self.offset = self.DEFAULT_OFFSET
        self.parking_threshold = self.DEFAULT_THRESHOLD
        self.detection_mode = "parking"  # Default detection mode
        self.log_data = []  # For logging events
        self.use_ml_detection = False
        self.ml_detector = None
        self.ml_confidence = self.DEFAULT_CONFIDENCE
        self._cleanup_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.video_lock = threading.Lock()

        # Initialize counters
        self.total_spaces = 0
        self.free_spaces = 0
        self.occupied_spaces = 0

        # Image dimensions
        self.image_width = 1280
        self.image_height = 720

        # Video sources - moved from detection_tab to here
        self.video_sources = list_available_videos()

        # Video reference map and dimensions
        self.setup_video_reference_map()
        self.current_reference_image = "carParkImg.png"  # Default

        # Load resources
        self.config_dir = "config"
        self.log_dir = "logs"
        ensure_directories_exist([self.config_dir, self.log_dir])
        self.load_parking_positions()

        # Initialize parking allocation components
        self.parking_visualizer = ParkingVisualizer(config_dir=self.config_dir, logs_dir=self.log_dir)
        self.allocation_engine = ParkingAllocationEngine(config_dir=self.config_dir)

        # Setup UI components
        self.setup_ui()

        # Start a monitoring thread to log data
        self.monitor_thread = threading.Thread(target=self.monitoring_thread, daemon=True)
        self.monitor_thread.start()

        self.master.bind("<Configure>", self.on_window_configure)
        # Create and connect the parking manager if not already created
        if not hasattr(self, 'parking_manager'):
            from models.parking_manager import ParkingManager
            self.parking_manager = ParkingManager(config_dir=self.config_dir, log_dir=self.log_dir)

        # Connect parking components
        self.parking_manager.parking_visualizer = self.parking_visualizer

        # After self.use_ml_detection initialization
        self.use_yolo_tracking = False
        self.vehicle_tracker = None

    def setup_video_reference_map(self):
        """Set up the map between videos and reference images"""
        self.video_reference_map = {
            "sample5.mp4": "saming1.png",
            "Video.mp4": "videoImg.png",
            "carPark.mp4": "carParkImg.png",
            "0": "webcamImg.png",  # Default for webcam
            "newVideo1.mp4": "newRefImage1.png",
            "newVideo2.mp4": "newRefImage2.png"
        }

        # Reference dimensions
        self.reference_dimensions = {
            "carParkImg.png": (1280, 720),
            "videoImg.png": (1280, 720),
            "webcamImg.png": (640, 480),
            "newRefImage1.png": (1280, 720),
            "newRefImage2.png": (1920, 1080)
        }

    def load_parking_positions(self, reference_image=None):
        """Load parking positions from file"""
        try:
            if reference_image is None:
                reference_image = self.current_reference_image

            # Load parking positions from file
            positions = load_parking_positions(self.config_dir, reference_image)

            # Store as original positions (at reference dimensions)
            self.original_posList = positions.copy()
            self.posList = positions.copy()  # Will be scaled below if needed

            self.total_spaces = len(self.posList)
            self.free_spaces = 0
            self.occupied_spaces = self.total_spaces

            # Only scale positions if dimensions are available
            if (reference_image in self.reference_dimensions and
                    hasattr(self, 'image_width') and
                    hasattr(self, 'image_height')):
                self.scale_positions_to_current_dimensions()

            # Update parking allocation module
            if hasattr(self, 'setup_tab'):
                self.setup_tab.update_allocation_data()
            elif hasattr(self, 'parking_visualizer') and hasattr(self, 'allocation_engine'):
                self.connect_parking_data()

            # Notify tabs of the updated positions
            if hasattr(self, 'detection_tab'):
                self.detection_tab.update_status_info(
                    self.total_spaces, self.free_spaces,
                    self.occupied_spaces, self.vehicle_counter
                )

        except Exception as e:
            self.log_event(f"Failed to load parking positions: {str(e)}")
            messagebox.showerror("Error", f"Failed to load parking positions: {str(e)}")
            self.total_spaces = 0
            self.free_spaces = 0
            self.occupied_spaces = 0

    def setup_ui(self):
        """Set up the application's user interface"""
        # Create main container
        self.main_container = ttk.Notebook(self.master)
        self.main_container.grid(row=0, column=0, sticky=NSEW, padx=5, pady=5)
        self.allocation_tab_frame = Frame(self.main_container)

        # Create tabs
        self.detection_tab_frame = Frame(self.main_container)
        self.setup_tab_frame = Frame(self.main_container)
        self.log_tab_frame = Frame(self.main_container)
        self.stats_tab_frame = Frame(self.main_container)
        self.reference_tab_frame = Frame(self.main_container)
        for frame in [self.detection_tab_frame, self.setup_tab_frame, self.log_tab_frame,
                      self.stats_tab_frame, self.reference_tab_frame, self.allocation_tab_frame]:
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)

        # Add tab frames to notebook
        self.main_container.add(self.detection_tab_frame, text="Detection")
        self.main_container.add(self.setup_tab_frame, text="Setup")
        self.main_container.add(self.log_tab_frame, text="Logs")
        self.main_container.add(self.stats_tab_frame, text="Statistics")
        self.main_container.add(self.reference_tab_frame, text="References")
        self.main_container.add(self.allocation_tab_frame, text="Parking Allocation")

        # Add tab selection event handler
        self.main_container.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Initialize tab objects
        self.detection_tab = DetectionTab(self.detection_tab_frame, self)
        self.setup_tab = SetupTab(self.setup_tab_frame, self)
        self.log_tab = LogTab(self.log_tab_frame, self)
        self.stats_tab = StatsTab(self.stats_tab_frame, self)
        self.reference_tab = ReferenceTab(self.reference_tab_frame, self)
        self.allocation_tab = ParkingAllocationTab(self.allocation_tab_frame, self)

        # Initialize parking data for allocation tab
        self.initialize_parking_allocation()

        #Add status bar at bottom
        self.status_bar = ttk.Label(self.master, text="Ready", relief="sunken")
        self.status_bar.grid(row=1, column=0, sticky=W + E)

    def log_event(self, message):
        """Log an event with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # Add to log data
        self.log_data.append(log_entry)

        # Update log display if it exists
        if hasattr(self, 'log_tab'):
            self.log_tab.add_log_entry(log_entry)

    def update_status_info(self):
        """Update status information across tabs"""
        if hasattr(self, 'detection_tab'):
            self.detection_tab.update_status_info(
                self.total_spaces,
                self.free_spaces,
                self.occupied_spaces,
                self.vehicle_counter
            )



    def monitoring_thread(self):
        """Background thread for monitoring and periodic logging"""
        while True:
            # Record stats every hour if detection is running
            if self.running and hasattr(self, 'stats_tab'):
                self.stats_tab.record_current_stats(
                    self.total_spaces,
                    self.free_spaces,
                    self.occupied_spaces,
                    self.vehicle_counter
                )

            # Sleep for an hour (3600 seconds)
            time.sleep(3600)

    # Modify the scale_positions_to_current_dimensions function in app.py
    # to consistently scale between reference and display dimensions

    def scale_positions_to_current_dimensions(self):
        """Scale parking positions based on current video dimensions - optimized"""
        try:
            if not hasattr(self, 'image_width') or not hasattr(self, 'image_height'):
                self.log_event("Cannot scale positions: image dimensions not set")
                return

            # Skip if no reference dimensions are available
            if self.current_reference_image not in self.reference_dimensions:
                self.log_event(f"No reference dimensions for {self.current_reference_image}")
                return

            # Get reference dimensions
            ref_width, ref_height = self.reference_dimensions[self.current_reference_image]

            # Make sure original_posList exists
            if not hasattr(self, 'original_posList') or not self.original_posList:
                self.original_posList = self.posList.copy()
                self.log_event(f"Created original_posList with {len(self.original_posList)} spaces")

            # Calculate scale factors
            width_scale = self.image_width / ref_width
            height_scale = self.image_height / ref_height

            # Scale from original positions to avoid cumulative scaling errors
            scaled_positions = []
            for x, y, w, h in self.original_posList:
                new_x = int(x * width_scale)
                new_y = int(y * height_scale)
                new_w = int(w * width_scale)
                new_h = int(h * height_scale)
                scaled_positions.append((new_x, new_y, new_w, new_h))

            # Replace current positions with scaled positions
            self.posList = scaled_positions
            self.log_event(f"Scaled {len(self.posList)} positions")
        except Exception as e:
            self.log_event(f"Error scaling positions: {str(e)}")

    def connect_parking_data(self):
        """Connect existing parking data with the new allocation system"""
        if hasattr(self, 'posList') and self.posList:
            # Make sure parking manager exists
            if not hasattr(self, 'parking_manager'):
                from models.parking_manager import ParkingManager
                self.parking_manager = ParkingManager(config_dir=self.config_dir, log_dir=self.log_dir)

            # Connect to visualizer
            self.parking_manager.parking_visualizer = self.parking_visualizer

            # Make sure allocation tab has access to allocation engine
            if hasattr(self, 'allocation_tab'):
                self.allocation_tab.allocation_engine = self.allocation_engine
                self.allocation_tab.app = self

                self.log_event("Connected parking data to allocation system")

            # Initialize parking spaces in the visualizer
            self.parking_visualizer.initialize_parking_spaces(self.posList)

            # Create a compatible data structure for the allocation engine
            spaces_data = {}
            for i, (x, y, w, h) in enumerate(self.posList):
                space_id = f"S{i + 1}"
                # Split spaces into sections based on position
                section = "A" if x < self.image_width / 2 else "B"
                section += "1" if y < self.image_height / 2 else "2"

                spaces_data[f"{space_id}-{section}"] = {
                    'position': (x, y, w, h),
                    'occupied': True,  # Default to occupied until detected as free
                    'vehicle_id': None,
                    'last_state_change': datetime.now(),
                    'distance_to_entrance': x + y,  # Simple distance estimation
                    'section': section
                }

            # Update the allocation engine's data structure
            self.allocation_engine.initialize_parking_spaces(spaces_data)
            self.log_event(f"Connected {len(self.posList)} parking spaces to allocation system")

    def on_closing(self):
        """Handle window closing event"""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.running = False
            if hasattr(self, 'video_capture') and self.video_capture:
                self.video_capture.release()
            self.master.destroy()

    def adjust_for_screen_size(self):
        """Adjust UI elements based on screen size"""
        width = self.master.winfo_width()
        height = self.master.winfo_height()

        # Adjust font sizes based on screen width
        if width < 1000:
            base_font_size = 9
        elif width < 1400:
            base_font_size = 10
        else:
            base_font_size = 11

        # Update fonts
        self.master.option_add('*Label.font', f'Arial {base_font_size}')
        self.master.option_add('*Button.font', f'Arial {base_font_size}')
        self.master.option_add('*Entry.font', f'Arial {base_font_size}')
        self.master.option_add('*Combobox.font', f'Arial {base_font_size}')

        # Log the adjustment
        self.log_event(f"UI adjusted for screen size: {width}x{height}")

    def on_window_configure(self, event):
        """Handle window resize events"""
        # Only process events from the main window
        if event.widget == self.master:
            # Avoid processing too many resize events
            if not hasattr(self, '_resize_timer'):
                self._resize_timer = None

            # Cancel previous timer
            if self._resize_timer:
                self.master.after_cancel(self._resize_timer)

            # Schedule adjustment after resize completes
            self._resize_timer = self.master.after(200, self.adjust_for_screen_size)

        # ... existing code ...
    def initialize_parking_allocation(self):
        """Initialize the parking allocation system"""
        try:
            if hasattr(self, 'parking_manager') and hasattr(self, 'allocation_tab'):
                # Make sure the allocation tab has access to the parking manager's data
                self.allocation_tab.app = self

                # Create the parking data structure if it doesn't exist
                if not hasattr(self.parking_manager, 'parking_data'):
                    self.parking_manager.parking_data = {}

                    # Initialize with parking spaces
                    for i, (x, y, w, h) in enumerate(self.posList):
                        # Generate section based on position
                        section = "A" if x < self.image_width / 2 else "B"
                        section += "1" if y < self.image_height / 2 else "2"

                        space_id = f"S{i + 1}-{section}"
                        self.parking_manager.parking_data[space_id] = {
                            'position': (x, y, w, h),
                            'occupied': True,  # Default to occupied until detected
                            'vehicle_id': None,
                            'last_state_change': datetime.now(),
                            'distance_to_entrance': x + y,  # Simple distance estimation
                            'section': section
                        }

                # Update allocation tab's UI
                self.allocation_tab.update_visualization()
                self.allocation_tab.update_statistics()

                self.log_event("Parking allocation system initialized")
        except Exception as e:
            self.log_event(f"Error initializing parking allocation: {str(e)}")

    def on_tab_changed(self, event):
        """Handle tab selection changes"""
        selected_tab = self.main_container.select()
        tab_text = self.main_container.tab(selected_tab, "text")

        # Log the tab change
        self.log_event(f"Tab changed to: {tab_text}")

        if tab_text == "Parking Allocation" and hasattr(self, 'allocation_tab'):
            # Update allocation tab when selected
            self.allocation_tab.on_tab_selected()
        elif tab_text == "Setup" and hasattr(self, 'setup_tab'):
            # Update setup tab when selected
            self.setup_tab.on_tab_selected() if hasattr(self.setup_tab, 'on_tab_selected') else None
        elif tab_text == "Detection" and hasattr(self, 'detection_tab'):
            # Update detection tab when selected
            self.detection_tab.on_tab_selected() if hasattr(self.detection_tab, 'on_tab_selected') else None

