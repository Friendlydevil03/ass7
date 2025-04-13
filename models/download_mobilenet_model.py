import os
import urllib.request


def download_model_files():
    """Download MobileNetSSD model files if they don't exist"""
    os.makedirs("models", exist_ok=True)

    # Prototxt file
    prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    prototxt_path = "models/MobileNetSSD_deploy.prototxt"

    # Caffemodel file
    caffemodel_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
    caffemodel_path = "models/MobileNetSSD_deploy.caffemodel"

    # Download files if they don't exist
    if not os.path.exists(prototxt_path):
        print(f"Downloading {prototxt_path}...")
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        print("Done!")

    if not os.path.exists(caffemodel_path):
        print(f"Downloading {caffemodel_path}...")
        urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
        print("Done!")


if __name__ == "__main__":
    download_model_files()