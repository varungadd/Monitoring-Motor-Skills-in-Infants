import yaml
import cv2
import numpy as np
import torch
from torchvision import transforms, utils
import math
import _init_paths
from models.adaptive_pose_hrnet import get_adaptive_pose_net
from core.inference import get_max_preds

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

def save_image_with_joints(image, keypoints, file_name):
    for (x, y) in keypoints:
        print(x, y)
        print(x*4, y*4)
        cv2.circle(image, (int(x*4), int(y*4)), 5, (0, 255, 0), -1)
    cv2.imwrite(file_name, image)

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    grid = utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def main(config_path, image_path, output_path):
    cfg = load_config(config_path)
    model_p, model_d = initialize_models(cfg)
    load_pretrained_weights(model_p, cfg['MODEL']['PRETRAINED'])
    input_image = preprocess_image(image_path, cfg['MODEL']['IMAGE_SIZE'])
    
    model_p.eval()
    with torch.no_grad():
        feature_output, keypoint_output = model_p(input_image)
        if isinstance(keypoint_output, list):
            keypoint_output = keypoint_output[-1]
            # feature_output = feature_keypoint_outputoutputs[-1]
        else:
            keypoint_output = keypoint_output
            feature_output = keypoint_output
    keypoints = get_keypoints_from_heatmap(keypoint_output, cfg['MODEL']['IMAGE_SIZE'])
    print(keypoints)
    
    # Load the original image for visualization
    image = cv2.imread(image_path)
    image_with_joints = image.copy()
    save_image_with_joints(image_with_joints, keypoints, output_path)

if __name__ == "__main__":
    config_path = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
    image_path = 'image.png'
    output_path = 'output_image_with_joints.png'
    main(config_path, image_path, output_path)