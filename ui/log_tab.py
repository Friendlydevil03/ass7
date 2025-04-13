from tkinter import Frame, Label, Button, Text, Scrollbar, messagebox
from tkinter import LEFT, RIGHT, BOTH, X, Y
from utils.resource_manager import save_log


class LogTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # Setup UI components
        self.setup_ui()

    def setup_ui(self):
        """Set up the log tab UI"""
        # Log tab frame
        self.log_frame = Frame(self.parent, padx=10, pady=10)
        self.log_frame.pack(fill=BOTH, expand=True)

        # Title and controls
        self.log_header = Frame(self.log_frame)
        self.log_header.pack(fill=X, pady=5)

        Label(self.log_header, text="System Logs", font=("Arial", 14, "bold")).pack(side=LEFT)

        self.clear_log_button = Button(self.log_header, text="Clear Log", command=self.clear_log)
        self.clear_log_button.pack(side=RIGHT, padx=5)

        self.save_log_button = Button(self.log_header, text="Save Log", command=self.save_log)
        self.save_log_button.pack(side=RIGHT, padx=5)

        # Log text area with scrollbar
        self.log_text_frame = Frame(self.log_frame)
        self.log_text_frame.pack(fill=BOTH, expand=True, pady=10)

        self.log_text = Text(self.log_text_frame, wrap="word", height=20)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)

        self.log_scrollbar = Scrollbar(self.log_text_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=RIGHT, fill=Y)

        self.log_text.config(yscrollcommand=self.log_scrollbar.set)
        self.log_text.config(state="disabled")

        # Add any existing logs
        for log_entry in self.app.log_data:
            self.add_log_entry(log_entry)

    def add_log_entry(self, entry):
        """Add an entry to the log text"""
        self.log_text.config(state="normal")
        self.log_text.insert("end", entry + "\n")
        self.log_text.see("end")  # Auto-scroll to the end
        self.log_text.config(state="disabled")

    def clear_log(self):
        """Clear the log display"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the log?"):
            self.log_text.config(state="normal")
            self.log_text.delete(1.0, "end")
            self.log_text.config(state="disabled")
            self.app.log_data = []
            self.app.log_event("Log cleared")

    def save_log(self):
        """Save the log to a file"""
        try:
            filename = save_log(self.app.log_data, self.app.log_dir)
            if filename:
                messagebox.showinfo("Success", f"Log saved to {filename}")
                self.app.log_event(f"Log saved to {filename}")
            else:
                messagebox.showerror("Error", "Failed to save log file")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {str(e)}")