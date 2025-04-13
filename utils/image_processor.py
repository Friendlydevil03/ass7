import cv2
import numpy as np
from PIL import Image, ImageTk


def preprocess_frame_for_parking_detection(img):
    """Preprocess a frame for parking space detection"""
    # Convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    # Apply threshold
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    # Apply blur again to smooth edges
    imgBlur = cv2.medianBlur(imgThreshold, 5)
    # Dilate to fill in holes
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgBlur, kernel, iterations=1)
    return imgDilate


def process_parking_spaces(img_pro, img, pos_list, threshold, debug=False):
    """Process and mark parking spaces in the image - optimized version"""
    space_counter = 0
    
    # Create a copy of img only if needed for drawing
    if len(pos_list) > 0:
        img_display = img  # Use direct reference to avoid copy unless needed
    else:
        return img, 0, 0, 0  # Return early if no positions
    
    # Precompute font and colors to avoid recreation
    font = cv2.FONT_HERSHEY_SIMPLEX
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)
    yellow_color = (255, 255, 0)

    # Add debug info
    if debug:
        img_height, img_width = img.shape[:2]
        cv2.putText(img_display, f"Image size: {img_width}x{img_height}", (10, 20),
                    font, 0.5, yellow_color, 1)

    for i, (x, y, w, h) in enumerate(pos_list):
        # Ensure coordinates are within image bounds and are integers
        if (y >= 0 and y + h < img_pro.shape[0] and x >= 0 and x + w < img_pro.shape[1]):
            # Convert coordinates to integers to avoid the error
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Add box number and coordinates in debug mode
            if debug:
                coord_text = f"Box {i}: ({x},{y})"
                cv2.putText(img_display, coord_text, (x, y - 5),
                            font, 0.4, yellow_color, 1)

            # Only extract the crop we need
            img_crop = img_pro[y:y + h, x:x + w]
            
            # Optimize counting - use sum instead of countNonZero for better performance
            count = np.sum(img_crop > 0)

            if count < threshold:
                color = green_color  # Green for free
                space_counter += 1
            else:
                color = red_color  # Red for occupied

            # Draw ID number for each space
            cv2.putText(img_display, str(i), (x + 5, y + 15),
                        font, 0.5, yellow_color, 2)
                        
            # Draw rectangle and count
            cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_display, str(count), (x, y + h - 3), font,
                        0.5, color, 2)

    free_spaces = space_counter
    total_spaces = len(pos_list)
    occupied_spaces = total_spaces - free_spaces

    return img_display, free_spaces, occupied_spaces, total_spaces


def detect_vehicles_traditional(current_frame, prev_frame, line_height, min_contour_width, min_contour_height, offset,
                                matches, vehicles_count):
    """
    Detect vehicles using traditional computer vision - optimized version
    """
    # Only create a copy of the frame if we need to draw on it
    display_frame = current_frame.copy()

    # Calculate absolute difference between frames
    d = cv2.absdiff(prev_frame, current_frame)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    # Apply blur and threshold
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Apply dilation and morphology operations
    # Optimize by combining operations when possible
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(cv2.dilate(th, np.ones((3, 3))), cv2.MORPH_CLOSE, kernel)

    # Find contours - use EXTERNAL type for faster processing
    contours, h = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make a copy of matches only if needed (if we have contours)
    if not contours:
        # Draw detection line
        cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Total Vehicle Detected: {vehicles_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)
        return display_frame, matches, vehicles_count

    matches_copy = matches.copy()

    # Draw detection line
    cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)

    # Process each contour - reduce the number processed if there are too many
    max_contours = 50  # Maximum contours to process for performance
    for (i, c) in enumerate(contours[:max_contours]):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        if not contour_valid:
            continue

        # Draw rectangle around vehicle
        cv2.rectangle(display_frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

        # Calculate centroid
        cx = x + w // 2
        cy = y + h // 2
        centroid = (cx, cy)

        # Add centroid to matches list
        matches_copy.append(centroid)

        # Draw centroid
        cv2.circle(display_frame, centroid, 5, (0, 255, 0), -1)

    # Count vehicles crossing the line
    new_vehicles_count = vehicles_count
    new_matches = []

    for (x, y) in matches_copy:
        # Check if centroid is near the line
        if line_height - offset < y < line_height + offset:
            new_vehicles_count += 1
        else:
            # Keep centroids that haven't crossed the line
            new_matches.append((x, y))

    # Display vehicle count
    cv2.putText(display_frame, f"Total Vehicle Detected: {new_vehicles_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)

    return display_frame, new_matches, new_vehicles_count


def process_ml_detections(frame, detections, line_height, offset, matches, vehicles_count, class_names):
    """Process detections from ML model - optimized version"""
    display_frame = frame.copy()

    # Draw detection line
    cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)

    # Make a deep copy of matches list
    matches_copy = matches.copy() if matches is not None else []

    # Handle case where detections might be None
    if detections is None:
        detections = []

    # Process only a limited number of detections for performance
    max_detections = 30
    for i, detection in enumerate(detections[:max_detections]):
        # Ensure detection has the expected format
        if len(detection) < 3:  # We need at least box, score, label
            continue

        box, score, label = detection[:3]  # Unpack the first 3 elements

        # Ensure box is valid
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = box

        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroid = (cx, cy)

        # Only add label if score is high enough (optimization)
        if score > 0.6:
            class_name = class_names[label] if label < len(class_names) else f"Class {label}"
            cv2.putText(display_frame, f"{class_name}: {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add centroid
        matches_copy.append(centroid)

        # Draw centroid
        cv2.circle(display_frame, centroid, 5, (0, 0, 255), -1)

    # Count vehicles crossing the line
    new_vehicles_count = vehicles_count
    new_matches = []

    for (x, y) in matches_copy:
        if line_height - offset < y < line_height + offset:
            new_vehicles_count += 1
        else:
            new_matches.append((x, y))

    # Display vehicle count
    cv2.putText(display_frame, f"Total Vehicle Detected: {new_vehicles_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)

    # Return all required values
    return display_frame, new_matches, new_vehicles_count
