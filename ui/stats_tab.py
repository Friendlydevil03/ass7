from datetime import datetime
from tkinter import Frame, Label, Button, ttk, messagebox
from tkinter import LEFT, RIGHT, BOTH, X, Y
from utils.resource_manager import export_statistics


class StatsTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # Setup UI components
        self.setup_ui()

    def setup_ui(self):
        """Set up the statistics tab UI"""
        # Stats tab frame
        self.stats_frame = Frame(self.parent, padx=10, pady=10)
        self.stats_frame.pack(fill=BOTH, expand=True)

        # Title
        Label(self.stats_frame, text="Parking Statistics", font=("Arial", 16, "bold")).pack(pady=10)

        # Statistics data
        self.stats_data_frame = Frame(self.stats_frame)
        self.stats_data_frame.pack(fill=BOTH, expand=True, pady=10)

        # Create Treeview for statistics
        self.stats_tree = ttk.Treeview(self.stats_data_frame,
                                       columns=("timestamp", "total", "free", "occupied", "vehicles"))

        # Define column headings
        self.stats_tree.heading("#0", text="")
        self.stats_tree.heading("timestamp", text="Timestamp")
        self.stats_tree.heading("total", text="Total Spaces")
        self.stats_tree.heading("free", text="Free Spaces")
        self.stats_tree.heading("occupied", text="Occupied Spaces")
        self.stats_tree.heading("vehicles", text="Vehicles Counted")

        # Define column widths
        self.stats_tree.column("#0", width=0, stretch=False)
        self.stats_tree.column("timestamp", width=200)
        self.stats_tree.column("total", width=100)
        self.stats_tree.column("free", width=100)
        self.stats_tree.column("occupied", width=100)
        self.stats_tree.column("vehicles", width=120)

        # Add scrollbar to treeview
        self.stats_vsb = ttk.Scrollbar(self.stats_data_frame, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=self.stats_vsb.set)
        self.stats_vsb.pack(side=RIGHT, fill=Y)
        self.stats_tree.pack(side=LEFT, fill=BOTH, expand=True)

        # Statistics controls
        self.stats_control_frame = Frame(self.stats_frame)
        self.stats_control_frame.pack(fill=X, pady=10)

        Button(self.stats_control_frame, text="Clear Statistics", command=self.clear_statistics).pack(side=RIGHT,
                                                                                                      padx=5)
        Button(self.stats_control_frame, text="Export Statistics", command=self.export_statistics).pack(side=RIGHT,
                                                                                                        padx=5)
        Button(self.stats_control_frame, text="Record Current Stats", command=self.record_current_stats).pack(
            side=RIGHT, padx=5)

    def record_current_stats(self, total_spaces=None, free_spaces=None, occupied_spaces=None, vehicle_counter=None):
        """Record current statistics to the stats view"""
        # Use provided values or get from app
        if total_spaces is None:
            total_spaces = self.app.total_spaces
        if free_spaces is None:
            free_spaces = self.app.free_spaces
        if occupied_spaces is None:
            occupied_spaces = self.app.occupied_spaces
        if vehicle_counter is None:
            vehicle_counter = self.app.vehicle_counter

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Insert at the beginning of the treeview
        self.stats_tree.insert("", 0, values=(
            timestamp,
            total_spaces,
            free_spaces,
            occupied_spaces,
            vehicle_counter
        ))

        self.app.log_event("Recorded current statistics")

    def clear_statistics(self):
        """Clear the statistics view"""
        if messagebox.askokcancel("Confirm", "Are you sure you want to clear all statistics?"):
            for item in self.stats_tree.get_children():
                self.stats_tree.delete(item)
            self.app.log_event("Statistics cleared")

    def export_statistics(self):
        """Export statistics to a CSV file"""
        try:
            # Collect data from treeview
            stats_data = []
            for item in self.stats_tree.get_children():
                values = self.stats_tree.item(item)["values"]
                stats_data.append(values)

            if not stats_data:
                messagebox.showinfo("Info", "No statistics data to export.")
                return

            # Export using the utility function
            filename = export_statistics(stats_data, self.app.log_dir)

            if filename:
                messagebox.showinfo("Success", f"Statistics exported to {filename}")
                self.app.log_event(f"Statistics exported to {filename}")
            else:
                messagebox.showerror("Error", "Failed to export statistics")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")