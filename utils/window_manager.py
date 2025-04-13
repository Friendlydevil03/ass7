import os
import json


class WindowManager:
    def __init__(self, root, config_dir, default_size="1280x720"):
        self.root = root
        self.config_file = os.path.join(config_dir, "window_config.json")
        self.default_size = default_size

    def save_window_position(self):
        """Save current window position and size"""
        try:
            # Get current window geometry
            geometry = self.root.geometry()

            # Create config directory if needed
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            # Save to file
            with open(self.config_file, "w") as f:
                json.dump({"geometry": geometry}, f)

        except Exception as e:
            print(f"Error saving window configuration: {str(e)}")

    def restore_window_position(self):
        """Restore previous window position and size"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    if "geometry" in config:
                        self.root.geometry(config["geometry"])
                    else:
                        self.root.geometry(self.default_size)
            else:
                # Use default size
                self.root.geometry(self.default_size)

        except Exception as e:
            print(f"Error restoring window configuration: {str(e)}")
            self.root.geometry(self.default_size)