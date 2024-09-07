import yaml
import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision
import math
import time
import _init_paths
from models.adaptive_pose_hrnet import get_adaptive_pose_net

def load_config(config_path):
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def initialize_models(cfg):
    model_p, model_d = get_adaptive_pose_net(cfg, is_train=False)
    return model_p, model_d

def load_pretrained_weights(model_p, pretrained_model_path):
    model_p.init_weights(pretrained_model_path)

def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def get_keypoints_from_heatmap(heatmap, image_size):
    heatmap = heatmap.squeeze().cpu().numpy()
    keypoints = []
    for i in range(heatmap.shape[0]):
        hmap = heatmap[i]
        idx = np.unravel_index(np.argmax(hmap), hmap.shape)
        y, x = idx
        keypoints.append((x * (image_size[1] / hmap.shape[1]), y * (image_size[0] / hmap.shape[0])))
    return keypoints

def save_image_with_joints(image, keypoints, output_file):
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), 2, [255, 0, 0], 2)
    cv2.imwrite(output_file, image)

def main(config_path, image_path, output_file):
    cfg = load_config(config_path)
    model_p, _ = initialize_models(cfg)
    load_pretrained_weights(model_p, cfg['MODEL']['PRETRAINED'])
    input_image = preprocess_image(image_path, cfg['MODEL']['IMAGE_SIZE'])
    
    model_p.eval()
    with torch.no_grad():
        feature_output, keypoint_output = model_p(input_image)
    
    keypoints = get_keypoints_from_heatmap(keypoint_output, cfg['MODEL']['IMAGE_SIZE'])
    print(keypoints)
    
    # Load the original image to draw keypoints
    original_image = cv2.imread(image_path)
    save_image_with_joints(original_image, keypoints, output_file)

if __name__ == "__main__":
    config_path = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
    image_path = 'image2.jpg'
    output_file = 'output_image_with_keypoints.png'
    main(config_path, image_path, output_file)
