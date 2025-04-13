import os

# Base paths
MEDIA_DIR = "media"
VIDEO_DIR = os.path.join(MEDIA_DIR, "videos")
REF_IMG_DIR = os.path.join(MEDIA_DIR, "references")


def ensure_media_dirs():
    """Ensure media directories exist"""
    for directory in [MEDIA_DIR, VIDEO_DIR, REF_IMG_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def get_video_path(video_name):
    """Get full path to a video file"""
    # Handle webcam case
    if video_name == "0":
        return 0

    # Check if video exists directly in root directory (for backward compatibility)
    if os.path.exists(video_name):
        return video_name

    # Check in videos directory
    video_path = os.path.join(VIDEO_DIR, video_name)
    if os.path.exists(video_path):
        return video_path

    # If not found, return original name and let OpenCV handle the error
    return video_name


def get_reference_image_path(image_name):
    """Get full path to a reference image file"""
    # Check if image exists directly in root directory (for backward compatibility)
    if os.path.exists(image_name):
        return image_name

    # Check in references directory
    img_path = os.path.join(REF_IMG_DIR, image_name)
    if os.path.exists(img_path):
        return img_path

    # If not found, return original name and let the application handle the error
    return image_name


def list_available_videos():
    """List all available video files in the videos directory"""
    ensure_media_dirs()

    videos = ["0"]  # Always include webcam option

    # Check root directory (for backward compatibility)
    try:
        root_videos = [f for f in os.listdir() if f.endswith(('.mp4', '.avi', '.mov'))]
        videos.extend(root_videos)
    except Exception as e:
        print(f"Error listing videos in root directory: {str(e)}")

    # Check videos directory
    try:
        if os.path.exists(VIDEO_DIR):
            dir_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
            videos.extend(dir_videos)
    except Exception as e:
        print(f"Error listing videos in videos directory: {str(e)}")

    # Make sure we have at least the webcam option
    if not videos:
        videos = ["0"]

    return videos

def list_available_references():
    """List all available reference images in the references directory"""
    ensure_media_dirs()

    references = []

    # Check root directory (for backward compatibility)
    root_images = [f for f in os.listdir() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    references.extend(root_images)

    # Check references directory
    if os.path.exists(REF_IMG_DIR):
        dir_images = [f for f in os.listdir(REF_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        references.extend(dir_images)

    return references