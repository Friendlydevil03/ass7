import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xgboost as xgb
import pandas as pd
from datetime import datetime
import pickle
import os


class ParkingVisualizer:
    def __init__(self, config_dir="config", logs_dir="logs"):
        self.config_dir = config_dir
        self.logs_dir = logs_dir

        # Ensure directories exist
        self._ensure_directories()

        # AI model for parking allocation
        self.model = None
        self.load_model()

        # Parking data
        self.parking_data = {}
        self.allocation_history = []

        # For visualization
        self.plot_width = 800
        self.plot_height = 600
        self.margin = 50
        self.space_width = 80
        self.space_height = 120

    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        for directory in [self.config_dir, self.logs_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        model_dir = os.path.join(self.config_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def load_model(self):
        """Load XGBoost model for parking allocation from file or create a new one"""
        model_path = os.path.join(self.config_dir, "models", "parking_allocation_model.pkl")

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Loaded existing XGBoost model")
            except Exception as e:
                print(f"Error loading model: {str(e)}. Creating a new one.")
                self._create_new_model()
        else:
            self._create_new_model()

    def _create_new_model(self):
        """Create and initialize a new XGBoost model"""
        # Initialize XGBoost model with parking-specific parameters
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            seed=42
        )

        # Create initial training data with basic features
        # This will be refined as the system collects more data
        initial_data = {
            'distance_to_entrance': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'time_since_last_occupied': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            'vehicle_size': [1, 2, 3, 1, 2, 3, 1, 2, 3, 2],
            'optimal_allocation': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Initial labels
        }

        df = pd.DataFrame(initial_data)
        X = df.drop('optimal_allocation', axis=1)
        y = df['optimal_allocation']

        # Train the model with initial data
        self.model.fit(X, y)

        # Save the model
        self.save_model()

    def save_model(self):
        """Save the XGBoost model to file"""
        try:
            model_path = os.path.join(self.config_dir, "models", "parking_allocation_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def update_model(self, new_data):
        """Update model with new training data"""
        if not new_data or len(new_data.get('distance_to_entrance', [])) == 0:
            print("No new data to update model")
            return

        df = pd.DataFrame(new_data)
        X = df.drop('optimal_allocation', axis=1)
        y = df['optimal_allocation']

        # Update model with new data
        self.model.fit(X, y)

        # Save updated model
        self.save_model()

        print("Model updated successfully with new data")

    def initialize_parking_spaces(self, positions):
        """Initialize parking space data structure from positions list"""
        self.parking_data = {}

        for i, (x, y, w, h) in enumerate(positions):
            space_id = f"S{i + 1}"

            # Calculate distance from entrance (simplified: using position as proxy)
            distance = x + y  # Simple proxy for distance

            self.parking_data[space_id] = {
                'id': space_id,
                'position': (x, y, w, h),
                'occupied': True,  # Default to occupied until detected as free
                'vehicle_id': None,
                'last_state_change': datetime.now(),
                'distance_to_entrance': distance,
                'allocation_score': 0,
                'occupation_history': []
            }

    def update_parking_status(self, space_ids, statuses):
        """Update the occupancy status of parking spaces"""
        current_time = datetime.now()

        for space_id, is_occupied in zip(space_ids, statuses):
            if space_id in self.parking_data:
                # If status changed, update history
                if self.parking_data[space_id]['occupied'] != is_occupied:
                    self.parking_data[space_id]['last_state_change'] = current_time

                    # Add to history
                    history_entry = {
                        'timestamp': current_time,
                        'state': 'occupied' if is_occupied else 'free',
                        'duration': (current_time - self.parking_data[space_id]['last_state_change']).total_seconds()
                    }
                    self.parking_data[space_id]['occupation_history'].append(history_entry)

                # Update status
                self.parking_data[space_id]['occupied'] = is_occupied

                # Clear vehicle if space is now free
                if not is_occupied:
                    self.parking_data[space_id]['vehicle_id'] = None

    def allocate_parking(self, vehicle_id, vehicle_size=1):
        """
        Allocate a vehicle to the optimal parking space using XGBoost model

        Parameters:
        - vehicle_id: Unique identifier for the vehicle
        - vehicle_size: Size category of vehicle (1=small, 2=medium, 3=large)

        Returns:
        - allocated_space_id: ID of the allocated parking space
        """
        # Get all free spaces
        free_spaces = [space_id for space_id, data in self.parking_data.items()
                       if not data['occupied']]

        if not free_spaces:
            return None  # No free spaces available

        # Prepare features for prediction
        current_time = datetime.now()
        features = []
        space_ids = []

        for space_id in free_spaces:
            space = self.parking_data[space_id]
            time_since_last_occupied = (current_time - space['last_state_change']).total_seconds() / 60  # minutes

            features.append([
                space['distance_to_entrance'],
                time_since_last_occupied,
                vehicle_size
            ])
            space_ids.append(space_id)

        # Convert to DataFrame for XGBoost prediction
        df = pd.DataFrame(features, columns=['distance_to_entrance',
                                             'time_since_last_occupied',
                                             'vehicle_size'])

        # Predict allocation scores
        scores = self.model.predict_proba(df)[:, 1]  # Probability of class 1 (optimal allocation)

        # Find best space based on scores
        best_idx = np.argmax(scores)
        best_space_id = space_ids[best_idx]
        best_score = scores[best_idx]

        # Update parking data
        self.parking_data[best_space_id]['occupied'] = True
        self.parking_data[best_space_id]['vehicle_id'] = vehicle_id
        self.parking_data[best_space_id]['allocation_score'] = best_score
        self.parking_data[best_space_id]['last_state_change'] = current_time

        # Log the allocation
        allocation = {
            'timestamp': current_time,
            'vehicle_id': vehicle_id,
            'space_id': best_space_id,
            'score': best_score,
            'vehicle_size': vehicle_size
        }
        self.allocation_history.append(allocation)

        return best_space_id

    def generate_visualization(self, save_path=None):
        """Generate a visual representation of the parking lot"""
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate grid dimensions based on number of spaces
        num_spaces = len(self.parking_data)
        cols = int(np.ceil(np.sqrt(num_spaces * 1.5)))  # Approximate width:height ratio of 3:2
        rows = int(np.ceil(num_spaces / cols))

        # Set plot limits
        ax.set_xlim(0, cols * self.space_width + self.margin * 2)
        ax.set_ylim(0, rows * self.space_height + self.margin * 2)

        # Draw parking spaces
        for i, (space_id, data) in enumerate(self.parking_data.items()):
            row = i // cols
            col = i % cols

            x = col * self.space_width + self.margin
            y = row * self.space_height + self.margin

            # Determine color based on occupancy
            if data['occupied']:
                color = 'red'
                alpha = 0.7
            else:
                color = 'green'
                alpha = 0.5

            # Draw the parking space rectangle
            rect = Rectangle((x, y), self.space_width - 5, self.space_height - 5,
                             linewidth=2, edgecolor='black', facecolor=color, alpha=alpha)
            ax.add_patch(rect)

            # Add space ID
            ax.text(x + 5, y + 5, space_id, fontsize=10, weight='bold')

            # Add vehicle ID if occupied
            if data['occupied'] and data['vehicle_id']:
                ax.text(x + 5, y + self.space_height - 20, f"V: {data['vehicle_id']}",
                        fontsize=8, color='white')

            # Add allocation score if applicable
            if data['allocation_score'] > 0:
                score_text = f"{data['allocation_score']:.2f}"
                ax.text(x + self.space_width - 30, y + 5, score_text, fontsize=8)

        # Add legend and title
        green_patch = Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.5)
        red_patch = Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7)
        ax.legend([green_patch, red_patch], ['Free', 'Occupied'], loc='upper right')

        # Add statistics
        free_count = sum(1 for data in self.parking_data.values() if not data['occupied'])
        total = len(self.parking_data)
        occupied_count = total - free_count

        ax.set_title(f'Parking Allocation Status - {free_count}/{total} Available')

        # Add timestamp
        ax.text(self.margin, rows * self.space_height + self.margin * 1.5,
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                fontsize=10)

        # Save the figure if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")

        return fig

    def mark_parking_spaces(self, frame, highlight_free=True):
        """Mark parking spaces on a video frame"""
        for space_id, data in self.parking_data.items():
            x, y, w, h = data['position']

            # Set color based on occupancy
            if data['occupied']:
                color = (0, 0, 255)  # Red for occupied
            else:
                color = (0, 255, 0)  # Green for free

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw space ID
            cv2.putText(frame, space_id, (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw vehicle ID if occupied
            if data['occupied'] and data['vehicle_id']:
                cv2.putText(frame, f"V:{data['vehicle_id']}", (x + 5, y + h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Highlight free spaces if requested
            if highlight_free and not data['occupied']:
                # Add a subtle highlight effect
                overlay = frame.copy()
                cv2.rectangle(overlay, (x + 2, y + 2), (x + w - 2, y + h - 2), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Add stats to frame
        free_count = sum(1 for data in self.parking_data.values() if not data['occupied'])
        total = len(self.parking_data)
        occupied_count = total - free_count

        cv2.putText(frame, f"Free: {free_count}/{total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Occupied: {occupied_count}/{total}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame