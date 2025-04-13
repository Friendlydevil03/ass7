import numpy as np
import pandas as pd
from datetime import datetime
import xgboost as xgb
import pickle
import os
import threading


class ParkingAllocationEngine:
    """
    AI-based parking allocation engine that uses XGBoost for optimal parking space allocation
    with load balancing capabilities.
    """

    def __init__(self, config_dir="config", model_file="parking_allocation_model.pkl"):

        self.config_dir = config_dir
        self.model_file = model_file
        self.model_path = os.path.join(config_dir, "models", model_file)

        # Create models directory if not exists
        model_dir = os.path.join(config_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Load or create the model
        self.model = self._load_or_create_model()

        # Track allocation history for adaptation
        self.allocation_history = []
        self.feedback_data = []

        # Track parking lot statistics for load balancing
        self.parking_stats = {}
        self.load_balancing_weight = 0.3  # How much weight to give to load balancing vs. convenience

        # Lock for thread safety
        self.lock = threading.Lock()

    def initialize_parking_spaces(self, spaces_data):
        with self.lock:
            # Store the parking spaces data
            self.parking_data = spaces_data

            # Update parking statistics for load balancing
            self.update_parking_stats(spaces_data)

    def _load_or_create_model(self):
        """Load existing model or create a new one if none exists"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading model: {e}. Creating a new one.")
                return self._create_new_model()
        else:
            return self._create_new_model()

    def _create_new_model(self):
        """Create and train a new XGBoost model with initial data"""
        # Create XGBoost classifier with optimized parameters
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1,
            random_state=42
        )

        # Create initial training data - with CORRECT feature names
        X = pd.DataFrame({
            'distance_to_entrance': [10, 20, 50, 30, 40, 60, 70, 25, 35, 45],
            'time_since_last_occupied': [60, 30, 15, 120, 45, 90, 10, 50, 75, 110],  # Changed from 'time_vacant'
            'vehicle_size': [1, 2, 3, 1, 2, 3, 1, 2, 2, 1]
            # Removed 'section_occupancy_rate' as it's not in the model
        })

        # Initial target labels (1 = good allocation)
        y = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Train model with initial data
        model.fit(X, y)

        # Save the model
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)

        return model

    def update_parking_stats(self, spaces_data):
        """
        Update the statistics about parking sections for load balancing

        Args:
            spaces_data: Dictionary of parking spaces with occupancy info
        """
        with self.lock:
            # Group spaces by section
            sections = {}
            for space_id, data in spaces_data.items():
                # Extract section from space_id (assuming format like "S1-A" where A is section)
                parts = space_id.split('-')
                section = parts[1] if len(parts) > 1 else 'A'  # Default to section A

                if section not in sections:
                    sections[section] = {'total': 0, 'occupied': 0}

                sections[section]['total'] += 1
                if data['occupied']:
                    sections[section]['occupied'] += 1

            # Calculate occupancy rates
            for section, stats in sections.items():
                if stats['total'] > 0:
                    occupancy_rate = stats['occupied'] / stats['total']
                else:
                    occupancy_rate = 0

                self.parking_stats[section] = {
                    'total': stats['total'],
                    'occupied': stats['occupied'],
                    'occupancy_rate': occupancy_rate
                }

    def get_section_from_space_id(self, space_id):
        """Extract section from space ID"""
        parts = space_id.split('-')
        return parts[1] if len(parts) > 1 else 'A'  # Default to section A

    def allocate_parking(self, spaces_data, vehicle_size=1, preferred_section=None):
        """
        Find the optimal parking space for a vehicle using XGBoost model and load balancing

        Args:
            spaces_data: Dictionary of parking spaces with their data
            vehicle_size: Size of vehicle (1=small, 2=medium, 3=large)
            preferred_section: Optional preferred parking section

        Returns:
            best_space_id: The ID of the optimal parking space
            allocation_score: The confidence score of the allocation
        """
        # Update parking statistics for load balancing - OUTSIDE the lock to avoid UI freezes
        self.update_parking_stats(spaces_data)

        # Filter for available spaces
        available_spaces = {space_id: data for space_id, data in spaces_data.items()
                            if not data['occupied']}

        if not available_spaces:
            return None, 0  # No available spaces

        # Prepare features for prediction
        current_time = datetime.now()
        features_list = []
        space_ids = []

        for space_id, data in available_spaces.items():
            # Calculate time since space became vacant (in minutes)
            time_since_last_occupied = (current_time - data['last_state_change']).total_seconds() / 60

            # Prepare features for this space - ONLY include the features the model expects!
            features = [
                data['distance_to_entrance'],
                time_since_last_occupied,  # Changed from 'time_vacant' to 'time_since_last_occupied'
                vehicle_size
            ]

            features_list.append(features)
            space_ids.append(space_id)

        # Only lock for the actual model prediction to minimize lock time
        with self.lock:
            # Convert to DataFrame with CORRECT feature names matching the model
            X_pred = pd.DataFrame(features_list, columns=[
                'distance_to_entrance',
                'time_since_last_occupied',  # Changed to match the model's expected name
                'vehicle_size'
            ])

            # Get prediction scores from model
            prediction_scores = self.model.predict_proba(X_pred)[:, 1]  # Probability of class 1

        # Apply load balancing adjustment
        balanced_scores = []
        for i, space_id in enumerate(space_ids):
            score = prediction_scores[i]

            # Get section
            section = self.get_section_from_space_id(space_id)

            # Apply load balancing adjustment based on section occupancy
            section_occupancy = self.parking_stats.get(section, {}).get('occupancy_rate', 0.5)
            load_balance_score = 1.0 - section_occupancy  # Higher score for less occupied sections

            # If preferred section is specified, boost its score
            section_preference = 1.0
            if preferred_section and section == preferred_section:
                section_preference = 1.2

            # Combine model score and load balancing with weightings
            balanced_score = (
                                     (1 - self.load_balancing_weight) * score +
                                     self.load_balancing_weight * load_balance_score
                             ) * section_preference

            balanced_scores.append(balanced_score)

        # Find best space
        if not balanced_scores:
            return None, 0

        best_idx = np.argmax(balanced_scores)
        best_space_id = space_ids[best_idx]
        best_score = balanced_scores[best_idx]

        # Log the allocation for model improvement
        allocation = {
            'timestamp': current_time,
            'space_id': best_space_id,
            'vehicle_size': vehicle_size,
            'score': best_score,
            'features': features_list[best_idx]
        }
        self.allocation_history.append(allocation)

        return best_space_id, best_score

    def add_feedback(self, space_id, vehicle_size, successful):
        """
        Add user feedback about allocation quality for model improvement

        Args:
            space_id: ID of the allocated space
            vehicle_size: Size of the vehicle
            successful: Whether the allocation was successful (user satisfaction)
        """
        with self.lock:
            # Get the section from the space ID
            section = self.get_section_from_space_id(space_id)

            # Find the allocation in history
            allocation = None
            for entry in reversed(self.allocation_history):
                if entry['space_id'] == space_id:
                    allocation = entry
                    break

            if allocation:
                feedback = {
                    'timestamp': datetime.now(),
                    'space_id': space_id,
                    'section': section,
                    'vehicle_size': vehicle_size,
                    'features': allocation['features'],
                    'successful': 1 if successful else 0
                }

                self.feedback_data.append(feedback)

                # Update model if we have enough feedback
                if len(self.feedback_data) >= 10:
                    self._retrain_model()

    def _retrain_model(self):
        """Retrain the model with collected feedback data"""
        if not self.feedback_data:
            return

        with self.lock:
            try:
                # Prepare features and labels
                features_list = []
                labels = []

                for feedback in self.feedback_data:
                    # Make sure to only include the first 3 features that the model expects
                    features_list.append(feedback['features'][:3])  # Only first 3 features
                    labels.append(feedback['successful'])

                # Convert to DataFrame with CORRECT feature names
                X = pd.DataFrame(features_list, columns=[
                    'distance_to_entrance',
                    'time_since_last_occupied',  # Changed from 'time_vacant'
                    'vehicle_size'
                ])
                y = np.array(labels)

                # Update model with new data
                self.model.fit(X, y)

                # Save updated model
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)

                # Clear feedback data
                self.feedback_data = []

                print("Model retrained with new feedback data")

            except Exception as e:
                print(f"Error retraining model: {e}")