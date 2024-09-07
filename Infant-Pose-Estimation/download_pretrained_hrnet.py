import os
import requests

# Define the URL and local path
url = "https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ"  # Replace with the actual URL
model_dir = os.path.join("models", "pytorch", "imagenet")
os.makedirs(model_dir, exist_ok=True)
local_path = os.path.join(model_dir, "hrnet_w48-8ef0771d.pth")

# Download the pre-trained model
response = requests.get(url, stream=True)   
with open(local_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print(f"Pre-trained model downloaded and saved to {local_path}")