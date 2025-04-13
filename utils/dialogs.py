from tkinter import Toplevel, StringVar, Frame, BOTH, X, LEFT, RIGHT, YES, NO
from tkinter import ttk


class CustomDialog:
    def __init__(self, parent, title, width=400, height=200):
        self.parent = parent
        self.result = None

        # Create the dialog window
        self.dialog = Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry(f"{width}x{height}")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.center_window(width, height)

        # Configure grid
        self.dialog.grid_columnconfigure(0, weight=1)
        self.dialog.grid_rowconfigure(0, weight=1)
        self.dialog.grid_rowconfigure(1, weight=0)

        # Content frame
        self.content_frame = ttk.Frame(self.dialog, padding=10)
        self.content_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Button frame
        self.button_frame = ttk.Frame(self.dialog)
        self.button_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=0)
        self.button_frame.columnconfigure(2, weight=0)

        # Standard buttons
        self.cancel_button = ttk.Button(self.button_frame, text="Cancel",
                                        command=self.cancel)
        self.cancel_button.grid(row=0, column=1, padx=5)

        self.ok_button = ttk.Button(self.button_frame, text="OK", style="Accent.TButton",
                                    command=self.ok)
        self.ok_button.grid(row=0, column=2, padx=5)

    def center_window(self, width, height):
        """Center the dialog on the parent window"""
        # Get parent geometry
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate position
        x = parent_x + (parent_width // 2) - (width // 2)
        y = parent_y + (parent_height // 2) - (height // 2)

        # Set position
        self.dialog.geometry(f"+{x}+{y}")

    def ok(self):
        """Handle OK button click"""
        self.validate_and_process()

    def cancel(self):
        """Handle Cancel button click"""
        self.dialog.destroy()

    def validate_and_process(self):
        """Override this method in subclasses to validate and process data"""
        # Default implementation just closes the dialog
        self.dialog.destroy()

    def show(self):
        """Show the dialog and wait for it to be closed"""
        self.dialog.wait_window()
        return self.result


class AssociateDialog(CustomDialog):
    def __init__(self, parent, app):
        super().__init__(parent, "Associate Video with Reference Image", 450, 250)
        self.app = app
        self.setup_content()

    def setup_content(self):
        """Setup the dialog content"""
        # Configure content frame
        for i in range(2):
            self.content_frame.columnconfigure(i, weight=1)

        # Video source selection
        ttk.Label(self.content_frame, text="Video Source:").grid(
            row=0, column=0, sticky="w", pady=(10, 5))

        self.video_var = StringVar(value=self.app.video_sources[0] if self.app.video_sources else "")
        self.video_dropdown = ttk.Combobox(self.content_frame, textvariable=self.video_var,
                                           values=self.app.video_sources, state="readonly")
        self.video_dropdown.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 5))

        # Reference image selection
        ttk.Label(self.content_frame, text="Reference Image:").grid(
            row=1, column=0, sticky="w", pady=(20, 5))

        from utils.media_paths import list_available_references
        available_refs = list_available_references()

        self.ref_var = StringVar(value=available_refs[0] if available_refs else "")
        self.ref_dropdown = ttk.Combobox(self.content_frame, textvariable=self.ref_var,
                                         values=available_refs, state="readonly")
        self.ref_dropdown.grid(row=1, column=1, sticky="ew", padx=10, pady=(20, 5))

    def validate_and_process(self):
        """Validate inputs and process the association"""
        video = self.video_var.get()
        ref_img = self.ref_var.get()

        if not video or not ref_img:
            from tkinter import messagebox
            messagebox.showerror("Error", "Please select both a video and reference image")
            return

        self.app.video_reference_map[video] = ref_img
        self.app.log_event(f"Associated video {video} with reference image {ref_img}")
        self.result = (video, ref_img)
        self.dialog.destroy()

# You can add more dialog classes as needed