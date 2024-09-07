import requests
import zipfile
import os
import shutil

# Define the URL and local file paths
url = "https://coe.northeastern.edu/Research/AClab/SyRIP/SyRIP.zip"
local_zip = "SyRIP.zip"
pose_root = "./"
syrip_dir = os.path.join(pose_root, "data", "syrip")
annotations_dir = os.path.join(syrip_dir, "annotations")
images_dir = os.path.join(syrip_dir, "images")

# Create necessary directories
os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(os.path.join(images_dir, "train_pre_infant"), exist_ok=True)
os.makedirs(os.path.join(images_dir, "train_infant"), exist_ok=True)
os.makedirs(os.path.join(images_dir, "validate_infant"), exist_ok=True)

# Download the file
response = requests.get(url, stream=True)
with open(local_zip, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

# Extract the zip file
with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(syrip_dir)

# Define source annotation paths
source_annotations = [
    ("SyRIP/annotations/1000S/person_keypoints_train_infant.json", "person_keypoints_train_pre_infant.json"),
    ("SyRIP/annotations/200R/person_keypoints_train_infant.json", "person_keypoints_train_infant.json"),
    ("SyRIP/annotations/validate100/person_keypoints_validate_infant.json", "person_keypoints_validate_infant.json"),
]


# Raise error if not exist
for src, dest in source_annotations:
    src_path = os.path.join(syrip_dir, src)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source annotation file not found: {src_path}")

# Move and rename annotation files
for src, dest in source_annotations:
    src_path = os.path.join(syrip_dir, src)
    dest_path = os.path.join(annotations_dir, dest)
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)

# Define image directories (update paths as needed if images are within subdirectories)
image_dirs = [
    ("SyRIP/images/train_pre_infant", os.path.join(images_dir, "train_pre_infant")),
    ("SyRIP/images/train_infant", os.path.join(images_dir, "train_infant")),
    ("SyRIP/images/validate_infant", os.path.join(images_dir, "validate_infant")),
]

# Move image files
for src, dest in image_dirs:
    src_path = os.path.join(syrip_dir, src)
    if os.path.exists(src_path):
        for item in os.listdir(src_path):
            shutil.move(os.path.join(src_path, item), os.path.join(dest, item))

print(f"Dataset downloaded and organized in {pose_root}/data/syrip")