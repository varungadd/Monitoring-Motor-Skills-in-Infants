import yaml
import cv2
import numpy as np
import torch
from torchvision import transforms
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

def visualize_keypoints(image_path, keypoints, image_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
    cv2.imshow('Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(config_path, image_path):
    cfg = load_config(config_path)
    model_p, model_d = initialize_models(cfg)
    load_pretrained_weights(model_p, cfg['MODEL']['PRETRAINED'])
    input_image = preprocess_image(image_path, cfg['MODEL']['IMAGE_SIZE'])
    
    model_p.eval()
    with torch.no_grad():
        feature_output, keypoint_output = model_p(input_image)
    print(keypoint_output)
    keypoints = get_keypoints_from_heatmap(keypoint_output, cfg['MODEL']['IMAGE_SIZE'])
    print("Feature:" , feature_output)
    print("Feature shape:", feature_output.shape)
    print("keypoints:", keypoints)
    print("Keypoint shape:", keypoint_output.shape)
    visualize_keypoints(image_path, keypoints, cfg['MODEL']['IMAGE_SIZE'])

if __name__ == "__main__":
    config_path = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
    image_path = 'image2.jpg'
    main(config_path, image_path)