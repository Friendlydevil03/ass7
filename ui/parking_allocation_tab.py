import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
import random
from datetime import datetime


class ParkingAllocationTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # Load components from other modules
        self.visualizer = self.app.parking_visualizer
        self.allocation_engine = self.app.allocation_engine

        # Set up UI state variables
        self.show_visualization = tk.BooleanVar(value=True)
        self.highlight_free_spaces = tk.BooleanVar(value=True)
        self.auto_allocation_enabled = tk.BooleanVar(value=False)
        self.preferred_section = tk.StringVar(value="Any")
        self.load_balancing_weight = tk.DoubleVar(value=0.3)
        self.vehicle_size = tk.IntVar(value=1)

        # Queue for thread-safe UI updates
        self.update_queue = queue.Queue()

        # Variables for vehicle simulation
        self.next_vehicle_id = 1
        self.allocated_vehicles = {}

        # Setup UI components
        self.setup_ui()

        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()

    def setup_ui(self):
        """Set up the UI components for the parking allocation tab"""
        # Configure grid layout
        self.parent.grid_columnconfigure(0, weight=3)  # Visualization area
        self.parent.grid_columnconfigure(1, weight=1)  # Control panel
        self.parent.grid_rowconfigure(0, weight=1)

        # Left side - Visualization area
        self.viz_frame = ttk.Frame(self.parent)
        self.viz_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(0, weight=1)

        # Canvas for visualization
        self.canvas_frame = ttk.Frame(self.viz_frame)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        # Initial visualization
        self.fig = plt.Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Right side - Control panel
        self.control_frame = ttk.Frame(self.parent)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Control panel sections
        self._add_visualization_controls()
        self._add_separator()
        self._add_allocation_controls()
        self._add_separator()
        self._add_stats_section()
        self._add_separator()
        self._add_simulation_controls()

    def _add_visualization_controls(self):
        """Add visualization controls to the control panel"""
        viz_frame = ttk.LabelFrame(self.control_frame, text="Visualization")
        viz_frame.pack(fill="x", padx=5, pady=5)

        # Toggle visualization view
        viz_check = ttk.Checkbutton(viz_frame, text="Show visualization",
                                    variable=self.show_visualization,
                                    command=self.update_visualization)
        viz_check.pack(anchor="w", padx=5, pady=3)

        # Highlight free spaces
        highlight_check = ttk.Checkbutton(viz_frame, text="Highlight free spaces",
                                          variable=self.highlight_free_spaces,
                                          command=self.update_visualization)
        highlight_check.pack(anchor="w", padx=5, pady=3)

        # Update button
        update_btn = ttk.Button(viz_frame, text="Refresh Visualization",
                                command=self.update_visualization)
        update_btn.pack(fill="x", padx=5, pady=5)

    def _add_allocation_controls(self):
        """Add allocation controls to the control panel"""
        alloc_frame = ttk.LabelFrame(self.control_frame, text="Space Allocation")
        alloc_frame.pack(fill="x", padx=5, pady=5)

        # Auto-allocation toggle
        auto_check = ttk.Checkbutton(alloc_frame, text="Enable auto allocation",
                                     variable=self.auto_allocation_enabled)
        auto_check.pack(anchor="w", padx=5, pady=3)

        # Section preference
        section_frame = ttk.Frame(alloc_frame)
        section_frame.pack(fill="x", padx=5, pady=3)

        ttk.Label(section_frame, text="Preferred section:").pack(side="left")
        section_combo = ttk.Combobox(section_frame, textvariable=self.preferred_section,
                                     values=["Any", "A", "B", "C", "D"])
        section_combo.pack(side="left", padx=5)

        # Vehicle size
        size_frame = ttk.Frame(alloc_frame)
        size_frame.pack(fill="x", padx=5, pady=3)

        ttk.Label(size_frame, text="Vehicle size:").pack(side="left")
        size_combo = ttk.Combobox(size_frame, textvariable=self.vehicle_size,
                                  values=[1, 2, 3])
        size_combo.pack(side="left", padx=5)

        # Load balancing weight
        lb_frame = ttk.Frame(alloc_frame)
        lb_frame.pack(fill="x", padx=5, pady=3)

        ttk.Label(lb_frame, text="Load balancing:").pack(side="left")
        lb_scale = ttk.Scale(lb_frame, from_=0.0, to=1.0, orient="horizontal",
                             variable=self.load_balancing_weight)
        lb_scale.pack(side="left", fill="x", expand=True, padx=5)

        # Allocate button
        allocate_btn = ttk.Button(alloc_frame, text="Allocate Vehicle",
                                  command=self.allocate_new_vehicle)
        allocate_btn.pack(fill="x", padx=5, pady=5)

    def _add_stats_section(self):
        """Add statistics section to the control panel"""
        stats_frame = ttk.LabelFrame(self.control_frame, text="Parking Statistics")
        stats_frame.pack(fill="x", padx=5, pady=5)

        # Stats display
        self.stats_text = tk.Text(stats_frame, height=6, width=30, wrap="word")
        self.stats_text.pack(fill="both", padx=5, pady=5)
        self.stats_text.insert("end", "Loading statistics...")
        self.stats_text.config(state="disabled")

    def _add_simulation_controls(self):
        """Add simulation controls to the control panel"""
        sim_frame = ttk.LabelFrame(self.control_frame, text="Simulation")
        sim_frame.pack(fill="x", padx=5, pady=5)

        # Simulation buttons
        add_vehicle_btn = ttk.Button(sim_frame, text="Add Random Vehicle",
                                     command=self.add_random_vehicle)
        add_vehicle_btn.pack(fill="x", padx=5, pady=3)

        remove_vehicle_btn = ttk.Button(sim_frame, text="Remove Random Vehicle",
                                        command=self.remove_random_vehicle)
        remove_vehicle_btn.pack(fill="x", padx=5, pady=3)

        reset_btn = ttk.Button(sim_frame, text="Reset Simulation",
                               command=self.reset_simulation)
        reset_btn.pack(fill="x", padx=5, pady=3)

    def _add_separator(self):
        """Add a separator line to the control panel"""
        ttk.Separator(self.control_frame, orient="horizontal").pack(fill="x", padx=5, pady=10)

    def update_visualization(self):
        """Update the parking visualization"""
        if self.show_visualization.get():
            try:
                # Clear the figure
                self.ax.clear()

                # Get FRESH parking data directly from the parking manager
                parking_data = {}
                if hasattr(self.app, 'parking_manager'):
                    if hasattr(self.app.parking_manager, 'parking_data'):
                        parking_data = self.app.parking_manager.parking_data.copy()

                # If no data is available, show a message
                if not parking_data:
                    self.ax.text(0.5, 0.5, "No parking data available",
                                 ha='center', va='center', fontsize=14)
                    self.canvas.draw()
                    return

                # Ensure allocated vehicles are reflected in the visualization
                for vehicle_id, space_id in self.allocated_vehicles.items():
                    if space_id in parking_data:
                        parking_data[space_id]['occupied'] = True
                        parking_data[space_id]['vehicle_id'] = vehicle_id

                # Calculate statistics
                free_count = sum(1 for data in parking_data.values() if not data.get('occupied', True))
                total = len(parking_data)
                occupied_count = total - free_count

                # Calculate grid dimensions
                if total > 0:
                    cols = max(4, int(np.ceil(np.sqrt(total))))
                    rows = int(np.ceil(total / cols))
                else:
                    cols, rows = 4, 4

                # Set plot limits
                space_w, space_h = 100, 60
                margin = 20
                self.ax.set_xlim(0, cols * space_w + 2 * margin)
                self.ax.set_ylim(0, rows * space_h + 2 * margin)

                # Draw parking spaces
                for i, (space_id, data) in enumerate(parking_data.items()):
                    row = i // cols
                    col = i % cols

                    x = col * space_w + margin
                    y = (rows - row - 1) * space_h + margin  # Invert y for better visualization

                    # Choose color based on occupancy
                    color = 'red' if data.get('occupied', True) else 'green'
                    edgecolor = 'black'

                    # Create rectangle
                    rect = plt.Rectangle((x, y), space_w - 5, space_h - 5,
                                         linewidth=2, edgecolor=edgecolor,
                                         facecolor=color, alpha=0.6)
                    self.ax.add_patch(rect)

                    # Add space ID
                    self.ax.text(x + 5, y + space_h - 15, space_id,
                                 fontsize=8, weight='bold', color='white')

                    # Add vehicle ID if occupied
                    if data.get('occupied', True) and data.get('vehicle_id'):
                        self.ax.text(x + 5, y + 10, f"V: {data['vehicle_id']}",
                                     fontsize=8, color='white')

                # Add title and legend
                self.ax.set_title(f"Parking Status: {free_count}/{total} Available")
                green_patch = plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.6)
                red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.6)
                self.ax.legend([green_patch, red_patch], ['Free', 'Occupied'])

                # Remove axis ticks for cleaner look
                self.ax.set_xticks([])
                self.ax.set_yticks([])

                # Add timestamp with formatted time to show update frequency
                current_time = datetime.now().strftime('%H:%M:%S')
                self.ax.text(margin, margin / 2,
                             f"Updated: {current_time}",
                             fontsize=8)

                # Force a complete redraw of the canvas
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

            except Exception as e:
                print(f"Error updating visualization: {str(e)}")
                import traceback
                traceback.print_exc()
                # Add error message to visualization
                self.ax.clear()
                self.ax.text(0.5, 0.5, f"Visualization Error: {str(e)}",
                             ha='center', va='center', fontsize=10, color='red')
                self.canvas.draw()

    def update_statistics(self):
        """Update the statistics display"""
        try:
            # Get current parking data
            parking_data = {}
            if hasattr(self.app, 'parking_manager'):
                if hasattr(self.app.parking_manager, 'parking_data'):
                    parking_data = self.app.parking_manager.parking_data.copy()

            # Calculate statistics
            total_spaces = len(parking_data) if parking_data else 0
            free_spaces = sum(
                1 for data in parking_data.values() if data.get('occupied') == False) if parking_data else 0
            occupied_spaces = total_spaces - free_spaces

            # Calculate occupancy rate safely
            try:
                occupancy_rate = (occupied_spaces / total_spaces) * 100 if total_spaces > 0 else 0
            except (ZeroDivisionError, TypeError):
                occupancy_rate = 0
                print("Warning: Error calculating occupancy rate - using 0")

            # Count allocations
            allocations = len(self.allocated_vehicles)

            # Format statistics text
            stats_text = (
                f"Total Spaces: {total_spaces}\n"
                f"Free Spaces: {free_spaces}\n"
                f"Occupied Spaces: {occupied_spaces}\n"
                f"Occupancy Rate: {occupancy_rate:.1f}%\n"
                f"Active Allocations: {allocations}\n"
                f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
            )

            # Update text widget
            self.stats_text.config(state="normal")
            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("end", stats_text)
            self.stats_text.config(state="disabled")

        except Exception as e:
            print(f"Error updating statistics: {str(e)}")
            # Log the full stack trace for debugging
            import traceback
            traceback.print_exc()

    def allocate_new_vehicle(self):
        """Allocate a new vehicle to a parking space"""
        # Run in a separate thread to avoid blocking UI
        threading.Thread(target=self._allocate_vehicle_thread, daemon=True).start()

    def _allocate_vehicle_thread(self):
        """Thread-safe vehicle allocation"""
        try:
            # Get parking data from parking manager
            parking_data = {}
            if hasattr(self.app, 'parking_manager'):
                with threading.Lock():  # Use lock for thread safety
                    parking_data = self.app.parking_manager.parking_data.copy()

            if not parking_data:
                self.queue_function(lambda: messagebox.showerror(
                    "Error", "No parking data available. Setup parking spaces first."))
                return

            # Check if there are free spaces
            free_spaces = {space_id: data for space_id, data in parking_data.items()
                           if not data.get('occupied', True)}

            if not free_spaces:
                self.queue_function(lambda: messagebox.showinfo(
                    "No Free Spaces", "No free parking spaces available."))
                return

            # Get vehicle parameters (capture current values)
            vehicle_id = f"V{self.next_vehicle_id}"
            vehicle_size = self.vehicle_size.get()
            preferred_section = None if self.preferred_section.get() == "Any" else self.preferred_section.get()
            weight = self.load_balancing_weight.get()

            # Set load balancing weight
            if hasattr(self.allocation_engine, 'load_balancing_weight'):
                self.allocation_engine.load_balancing_weight = weight

            # Perform allocation
            best_space_id, score = self.allocation_engine.allocate_parking(
                parking_data, vehicle_size, preferred_section)

            if best_space_id:
                # Update parking data
                if best_space_id in parking_data:
                    # Update data in thread-safe manner
                    with threading.Lock():
                        if hasattr(self.app, 'parking_manager'):
                            self.app.parking_manager.parking_data[best_space_id]['occupied'] = True
                            self.app.parking_manager.parking_data[best_space_id]['vehicle_id'] = vehicle_id

                    # Store allocation
                    self.allocated_vehicles[vehicle_id] = best_space_id
                    self.next_vehicle_id += 1  # Increment counter

                    # Schedule UI updates in main thread
                    self.queue_function(lambda: messagebox.showinfo(
                        "Allocation Success",
                        f"Vehicle {vehicle_id} allocated to space {best_space_id}\n"
                        f"Allocation score: {score:.2f}"))

                    # Update UI
                    self.queue_function(self.update_visualization)
                    self.queue_function(self.update_statistics)
                else:
                    self.queue_function(lambda: messagebox.showerror(
                        "Allocation Error", f"Space {best_space_id} not found in parking data."))
            else:
                self.queue_function(lambda: messagebox.showinfo(
                    "Allocation Failed", "Could not find a suitable parking space."))

        except Exception as e:
            self.queue_function(lambda: messagebox.showerror(
                "Error", f"Allocation error: {str(e)}"))

    def _perform_allocation(self, vehicle_size, preferred_section, weight):
        """Worker function to perform allocation without freezing UI"""
        try:
            # Get parking data
            parking_data = self.app.parking_manager.parking_data if hasattr(self.app, 'parking_manager') else {}

            # Check if there are free spaces
            free_spaces = {space_id: data for space_id, data in parking_data.items()
                           if not data.get('occupied', True)}

            if not free_spaces:
                # Use queue_function to show message in main thread
                self.queue_function(
                    lambda: messagebox.showinfo("No Free Spaces", "No free parking spaces available.")
                )
                return

            # Get vehicle parameters - use the passed parameters, not UI variables
            vehicle_id = f"V{self.next_vehicle_id}"

            # Set load balancing weight safely
            if hasattr(self.allocation_engine, 'load_balancing_weight'):
                self.allocation_engine.load_balancing_weight = weight

            # Perform allocation
            best_space_id, score = self.allocation_engine.allocate_parking(
                parking_data, vehicle_size, preferred_section)

            # Schedule UI updates and messages in the main thread
            if best_space_id:
                # Update data structures first
                if best_space_id in parking_data:
                    parking_data[best_space_id]['occupied'] = True
                    parking_data[best_space_id]['vehicle_id'] = vehicle_id

                    # Store allocation safely with mutex if needed
                    self.allocated_vehicles[vehicle_id] = best_space_id
                    # Update ID counter safely
                    self.next_vehicle_id += 1

                    # Schedule UI updates for main thread
                    self.queue_function(
                        lambda: messagebox.showinfo("Allocation Success",
                                                    f"Vehicle {vehicle_id} allocated to space {best_space_id}\n"
                                                    f"Allocation score: {score:.2f}")
                    )
                    self.queue_function(self.update_visualization)
                    self.queue_function(self.update_statistics)
                else:
                    self.queue_function(
                        lambda: messagebox.showerror("Allocation Error",
                                                     f"Space {best_space_id} not found in parking data.")
                    )
            else:
                self.queue_function(
                    lambda: messagebox.showinfo("Allocation Failed",
                                                "Could not find a suitable parking space.")
                )

        except Exception as e:
            # Safely show error in main thread
            error_msg = str(e)
            self.queue_function(
                lambda: messagebox.showerror("Error", f"Allocation error: {error_msg}")
            )

    def add_random_vehicle(self):
        """Add a random vehicle for simulation purposes"""
        # Generate random vehicle parameters
        vehicle_size = random.randint(1, 3)
        self.vehicle_size.set(vehicle_size)

        # Random section preference (80% chance of "Any")
        if random.random() < 0.8:
            self.preferred_section.set("Any")
        else:
            self.preferred_section.set(random.choice(["A", "B", "C", "D"]))

        # Allocate the vehicle
        self.allocate_new_vehicle()

    def remove_random_vehicle(self):
        """Remove a randomly selected vehicle"""
        # Run in a separate thread to avoid blocking UI
        threading.Thread(target=self._remove_vehicle_thread, daemon=True).start()

    def _remove_vehicle_thread(self):
        """Thread-safe vehicle removal"""
        try:
            if not self.allocated_vehicles:
                self.queue_function(lambda: messagebox.showinfo(
                    "No Vehicles", "No vehicles are currently allocated."))
                return

            # Select a random vehicle
            vehicle_id = random.choice(list(self.allocated_vehicles.keys()))
            space_id = self.allocated_vehicles[vehicle_id]

            # Update parking data in thread-safe manner
            with threading.Lock():
                if hasattr(self.app, 'parking_manager'):
                    if space_id in self.app.parking_manager.parking_data:
                        self.app.parking_manager.parking_data[space_id]['occupied'] = False
                        self.app.parking_manager.parking_data[space_id]['vehicle_id'] = None

            # Remove from allocated vehicles
            del self.allocated_vehicles[vehicle_id]

            # Update UI in main thread
            self.queue_function(self.update_visualization)
            self.queue_function(self.update_statistics)
            self.queue_function(lambda: messagebox.showinfo(
                "Vehicle Removed", f"Vehicle {vehicle_id} removed from space {space_id}."))

        except Exception as e:
            self.queue_function(lambda: messagebox.showerror(
                "Error", f"Error removing vehicle: {str(e)}"))

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the simulation?")
        if not confirm:
            return

        # Reset parking data
        parking_data = self.app.parking_manager.parking_data if hasattr(self.app, 'parking_manager') else {}
        for space_id in parking_data:
            parking_data[space_id]['occupied'] = False
            parking_data[space_id]['vehicle_id'] = None

        # Clear allocated vehicles
        self.allocated_vehicles = {}
        self.next_vehicle_id = 1

        # Update UI
        self.update_visualization()
        self.update_statistics()

        messagebox.showinfo("Simulation Reset", "Simulation has been reset.")

    def update_loop(self):
        """Background thread for periodic updates"""
        update_interval = 5  # Update visualization every 5 seconds
        visualization_counter = 0

        while self.running:
            # Process any UI update events in queue
            try:
                while not self.update_queue.empty():
                    func, args = self.update_queue.get_nowait()
                    func(*args)
            except queue.Empty:
                pass

            # Perform auto-allocation if enabled
            if self.auto_allocation_enabled.get():
                # Add a vehicle occasionally
                if random.random() < 0.1:  # 10% chance each cycle
                    self.update_queue.put((self.add_random_vehicle, ()))

                # Remove a vehicle occasionally
                if self.allocated_vehicles and random.random() < 0.05:  # 5% chance each cycle
                    self.update_queue.put((self.remove_random_vehicle, ()))

            # Update visualization only every X seconds
            visualization_counter += 1
            if visualization_counter >= update_interval:
                # Schedule a visualization update
                self.update_queue.put((self.update_visualization, ()))
                # Statistics every 10 seconds
                if visualization_counter >= update_interval * 2:
                    self.update_queue.put((self.update_statistics, ()))
                    visualization_counter = 0

            # Sleep between updates (1 second to keep other operations responsive)
            time.sleep(1)

    def queue_function(self, func, *args):
        """Queue a function to be executed in the main thread"""
        if not hasattr(self, 'update_queue'):
            self.update_queue = queue.Queue()
        self.update_queue.put((func, args))

    def on_tab_selected(self):
        """Called when this tab is selected"""
        print("Parking allocation tab selected")
        self.ensure_parking_data()
        self.update_visualization()
        self.update_statistics()

    def ensure_parking_data(self):
        """Ensure we have proper parking data to visualize"""
        if hasattr(self.app, 'parking_manager') and not hasattr(self.app.parking_manager, 'parking_data'):
            self.app.parking_manager.parking_data = {}

            # Initialize from positions
            if hasattr(self.app, 'posList') and self.app.posList:
                for i, (x, y, w, h) in enumerate(self.app.posList):
                    space_id = f"S{i + 1}"
                    section = "A" if x < self.app.image_width / 2 else "B"
                    section += "1" if y < self.app.image_height / 2 else "2"
                    full_space_id = f"{space_id}-{section}"

                    self.app.parking_manager.parking_data[full_space_id] = {
                        'position': (x, y, w, h),
                        'occupied': True,  # Default to occupied
                        'vehicle_id': None,
                        'last_state_change': datetime.now(),
                        'distance_to_entrance': x + y,
                        'section': section
                    }

    def cleanup(self):
        """Clean up resources before closing"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)

    # Add this method to your allocation class (if it exists)

    def allocate_group(self, vehicle_id, vehicle_size):
        """Allocate a vehicle to a group of spaces"""
        # Get all free groups
        free_groups = {space_id: data for space_id, data in self.parking_data.items()
                       if data.get('is_group', False) and not data['occupied']}

        if not free_groups:
            return None, 0  # No free groups

        # Find the best group based on size matching
        best_group = None
        best_score = 0

        for group_id, data in free_groups.items():
            # Get number of spaces in this group
            member_count = len(data.get('member_spaces', []))

            # Score based on size matching (higher is better)
            size_match = 1 - abs(member_count - vehicle_size) / max(member_count, vehicle_size)

            # Consider distance to entrance
            distance_score = 1 / (1 + data.get('distance_to_entrance', 0) / 1000)

            # Combine scores
            score = (size_match * 0.7) + (distance_score * 0.3)

            if score > best_score:
                best_score = score
                best_group = group_id

        return best_group, best_score