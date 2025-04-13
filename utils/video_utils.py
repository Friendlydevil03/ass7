"""
Utilities for handling video files and sources
"""
import os
import cv2
from pathlib import Path


def list_available_videos():
    """
    List all available video sources including webcam and video files in the videos directory

    Returns:
        list: List of video source names/paths
    """
    # Start with webcam
    sources = ["Webcam"]

    # Look for videos in common directories
    video_dirs = ["videos", "data/videos", "assets/videos"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    for video_dir in video_dirs:
        if os.path.exists(video_dir) and os.path.isdir(video_dir):
            for file in os.listdir(video_dir):
                filepath = os.path.join(video_dir, file)
                if os.path.isfile(filepath) and os.path.splitext(file)[1].lower() in video_extensions:
                    sources.append(filepath)

    # Also add sample videos in the project root
    for file in os.listdir("."):
        if os.path.isfile(file) and os.path.splitext(file)[1].lower() in video_extensions:
            sources.append(file)

    return sources


def get_video_dimensions(video_path):
    """
    Get the dimensions of a video

    Args:
        video_path: Path to video file or camera index (0 for webcam)

    Returns:
        tuple: (width, height) of the video, or (640, 480) as default if can't be determined
    """
    try:
        # Open video capture
        cap = cv2.VideoCapture(video_path)

        # Check if opened successfully
        if not cap.isOpened():
            return 640, 480

        # Get width and height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Release the video
        cap.release()

        return width, height

    except Exception as e:
        print(f"Error getting video dimensions: {str(e)}")
        return 640, 480  # Default dimensions


def check_camera_available(camera_index=0):
    """
    Check if a camera is available

    Args:
        camera_index: Camera index to check (default 0)

    Returns:
        bool: True if camera is available, False otherwise
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        available = cap.isOpened()
        cap.release()
        return available
    except:
        return False