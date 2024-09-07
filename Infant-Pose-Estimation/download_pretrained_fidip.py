import requests
import os

# Define the URLs and local file paths
urls = {
    "hrnet_fidip.pth": "https://drive.usercontent.google.com/download?id=19tBMoVS8wTza7VqVfPc6KBQ_POPAEIqb&export=download&authuser=0&confirm=t&uuid=c6b543f2-3339-475a-9006-c7cd8ff21f68&at=APZUnTX8hXEFWVydS0tlcVYA6R-0:1716911217100",
    "mobile_fidip.pth": "https://drive.usercontent.google.com/download?id=1aHaBwO3XBYtFXI7cDMRIzMWRffIJzrQU&export=download&authuser=0&confirm=t&uuid=383fbd25-0c41-407e-8f69-eab46c279abf&at=APZUnTVq78pObRTMhqQm5zkXiB13:1716911073666",
    "coco/posemobile.pth": "https://drive.usercontent.google.com/download?id=1TKvk0R_qrWV3UeSTawUYM5Burp-lOmss&export=download&authuser=0&confirm=t&uuid=25377a6c-3d3a-4ae5-bc11-92e0c8deaaed&at=APZUnTXrPozWhWqRbFqxeh5St_Wf:1716911504839"  # Replace with the actual URL
}

pose_root = "./"
models_dir = os.path.join(pose_root, "models")
coco_dir = os.path.join(models_dir, "coco")

# Create necessary directories
os.makedirs(models_dir, exist_ok=True)
os.makedirs(coco_dir, exist_ok=True)

# Download the files
for file_name, url in urls.items():
    local_path = os.path.join(models_dir if "coco/" not in file_name else coco_dir, os.path.basename(file_name))
    response = requests.get(url, stream=True)
    with open(local_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {local_path}")

print(f"Models downloaded and organized in {pose_root}/models")
