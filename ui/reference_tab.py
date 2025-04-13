import os
import cv2
from PIL import Image, ImageTk
from tkinter import Frame, Label, Button, Canvas, ttk, filedialog, messagebox
from tkinter import LEFT, RIGHT, BOTH, X, Y
from utils.media_paths import get_reference_image_path, list_available_references  # Add this import


class ReferenceTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # Setup UI components
        self.setup_ui()

    def setup_ui(self):
        """Set up a tab for reference image management"""
        # Reference tab frame
        self.reference_frame = Frame(self.parent, padx=10, pady=10)
        self.reference_frame.pack(fill=BOTH, expand=True)

        # Header frame
        header_frame = Frame(self.reference_frame)
        header_frame.pack(fill=X, pady=5)

        Label(header_frame, text="Reference Images", font=("Arial", 14, "bold")).pack(side=LEFT)

        # Add buttons
        Button(header_frame, text="Add Reference", command=self.browse_reference_image).pack(side=RIGHT, padx=5)
        Button(header_frame, text="Associate Video", command=self.associate_video_with_reference).pack(side=RIGHT,
                                                                                                       padx=5)

        # Create Treeview for references
        ref_tree_frame = Frame(self.reference_frame)
        ref_tree_frame.pack(fill=BOTH, expand=True, pady=10)

        self.ref_tree = ttk.Treeview(ref_tree_frame, columns=("image", "dimensions", "associated_videos"))

        # Define column headings
        self.ref_tree.heading("#0", text="")
        self.ref_tree.heading("image", text="Reference Image")
        self.ref_tree.heading("dimensions", text="Dimensions")
        self.ref_tree.heading("associated_videos", text="Associated Videos")

        # Define column widths
        self.ref_tree.column("#0", width=0, stretch=False)
        self.ref_tree.column("image", width=200)
        self.ref_tree.column("dimensions", width=150)
        self.ref_tree.column("associated_videos", width=300)

        # Add scrollbar
        ref_vsb = ttk.Scrollbar(ref_tree_frame, orient="vertical", command=self.ref_tree.yview)
        self.ref_tree.configure(yscrollcommand=ref_vsb.set)
        ref_vsb.pack(side=RIGHT, fill=Y)
        self.ref_tree.pack(side=LEFT, fill=BOTH, expand=True)

        # Preview frame
        preview_frame = Frame(self.reference_frame)
        preview_frame.pack(fill=BOTH, expand=True, pady=10)

        Label(preview_frame, text="Image Preview", font=("Arial", 12, "bold")).pack(pady=5)

        self.preview_canvas = Canvas(preview_frame, bg="black", height=300)
        self.preview_canvas.pack(fill=BOTH, expand=True)

        # Populate the reference tree
        self.populate_reference_tree()

        # Bind selection event
        self.ref_tree.bind("<<TreeviewSelect>>", self.on_reference_select)

    def populate_reference_tree(self):
        """Populate the reference image tree with data"""
        # Clear existing items
        for item in self.ref_tree.get_children():
            self.ref_tree.delete(item)

        # Add each reference image
        for ref_img in set(self.app.video_reference_map.values()):
            # Try to find the image first
            ref_img_path = ref_img
            if not os.path.exists(ref_img_path):
                ref_img_path = get_reference_image_path(ref_img)
                if not os.path.exists(ref_img_path):
                    continue  # Skip if image doesn't exist

            # Find associated videos
            associated = [vid for vid, img in self.app.video_reference_map.items() if img == ref_img]
            associated_str = ", ".join(associated)

            # Get dimensions
            dimensions = self.app.reference_dimensions.get(ref_img, "Unknown")
            if dimensions != "Unknown":
                dimensions_str = f"{dimensions[0]}x{dimensions[1]}"
            else:
                dimensions_str = "Unknown"

            # Insert into tree
            self.ref_tree.insert("", "end", values=(ref_img, dimensions_str, associated_str))

    def on_reference_select(self, event):
        """Handle reference image selection"""
        selection = self.ref_tree.selection()
        if selection:
            item = selection[0]
            ref_img = self.ref_tree.item(item, "values")[0]

            # Display the image in the preview canvas
            try:
                ref_img_path = get_reference_image_path(ref_img)
                if os.path.exists(ref_img_path):
                    img = cv2.imread(ref_img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Resize for preview
                        preview_height = 300
                        ratio = preview_height / img.shape[0]
                        preview_width = int(img.shape[1] * ratio)

                        img = cv2.resize(img, (preview_width, preview_height))

                        # Convert to PhotoImage
                        img_pil = Image.fromarray(img)
                        img_tk = ImageTk.PhotoImage(image=img_pil)

                        # Update canvas
                        self.preview_canvas.config(width=preview_width, height=preview_height)
                        self.preview_canvas.create_image(0, 0, anchor="nw", image=img_tk)
                        self.preview_canvas.image = img_tk
                    else:
                        self.app.log_event(f"Could not read image: {ref_img_path}")
                else:
                    self.app.log_event(f"Image does not exist: {ref_img_path}")
            except Exception as e:
                self.app.log_event(f"Error previewing reference image: {str(e)}")

    def browse_reference_image(self):
        """Browse for a new reference image and add it to the system"""
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )

        if file_path:
            # Get just the filename
            file_name = os.path.basename(file_path)

            # Determine target path in media/references directory
            from utils.media_paths import REF_IMG_DIR, ensure_media_dirs
            ensure_media_dirs()
            target_path = os.path.join(REF_IMG_DIR, file_name)

            # Copy the file to the references directory
            import shutil
            try:
                shutil.copy(file_path, target_path)
                self.app.log_event(f"Copied reference image {file_name} to references directory")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy reference image: {str(e)}")
                return

            # Get image dimensions
            try:
                img = cv2.imread(target_path)
                height, width = img.shape[:2]

                # Add to reference dimensions
                self.app.reference_dimensions[file_name] = (width, height)

                # Update the reference tree
                self.populate_reference_tree()

                # Select the new image
                for item in self.ref_tree.get_children():
                    if self.ref_tree.item(item, "values")[0] == file_name:
                        self.ref_tree.selection_set(item)
                        self.ref_tree.see(item)
                        self.on_reference_select(None)
                        break

                self.app.log_event(f"Added reference image {file_name} ({width}x{height})")
                messagebox.showinfo("Success", f"Added reference image: {file_name}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process reference image: {str(e)}")
                self.app.log_event(f"Error processing reference image: {str(e)}")

    def associate_video_with_reference(self):
        """Associate a video source with a reference image"""
        from tkinter import Toplevel, StringVar

        # Check if video sources are available
        if not hasattr(self.app, 'video_sources') or not self.app.video_sources:
            messagebox.showerror("Error", "No video sources available")
            return

        # Create a simple dialog
        dialog = Toplevel(self.parent)
        dialog.title("Associate Video with Reference Image")
        dialog.geometry("400x200")
        dialog.resizable(False, False)

        # Video source selection
        Label(dialog, text="Video Source:").pack(pady=(10, 5))
        video_var = StringVar(value=self.app.video_sources[0])
        video_dropdown = ttk.Combobox(dialog, textvariable=video_var, values=self.app.video_sources)
        video_dropdown.pack(fill=X, padx=20, pady=5)

        # Reference image selection
        Label(dialog, text="Reference Image:").pack(pady=(10, 5))
        ref_var = StringVar(
            value=list(self.app.video_reference_map.values())[0] if self.app.video_reference_map else "")

        # Collect all possible reference images
        available_references = list_available_references()

        ref_dropdown = ttk.Combobox(dialog, textvariable=ref_var, values=available_references)
        ref_dropdown.pack(fill=X, padx=20, pady=5)

        # Button frame
        btn_frame = Frame(dialog)
        btn_frame.pack(fill=X, pady=20)

        # Cancel button
        Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=RIGHT, padx=20)

        # Associate button
        def do_associate():
            video = video_var.get()
            ref_img = ref_var.get()

            if not video or not ref_img:
                messagebox.showerror("Error", "Please select both a video and reference image")
                return

            self.app.video_reference_map[video] = ref_img
            self.app.log_event(f"Associated video {video} with reference image {ref_img}")

            # Update UI
            self.populate_reference_tree()
            messagebox.showinfo("Success", f"Associated {video} with {ref_img}")
            dialog.destroy()

        Button(btn_frame, text="Associate", command=do_associate).pack(side=RIGHT, padx=20)