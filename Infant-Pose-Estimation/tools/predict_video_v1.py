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

def preprocess_frame(frame, image_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_size[1], image_size[0]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame = transform(frame).unsqueeze(0)
    return frame

def get_keypoints_from_heatmap(heatmap, image_size):
    heatmap = heatmap.squeeze().cpu().numpy()
    keypoints = []
    for i in range(heatmap.shape[0]):
        hmap = heatmap[i]
        idx = np.unravel_index(np.argmax(hmap), hmap.shape)
        y, x = idx
        keypoints.append((x * (image_size[1] / hmap.shape[1]), y * (image_size[0] / hmap.shape[0])))
    return keypoints

def draw_keypoints_on_frame(frame, keypoints):
    for (x, y) in keypoints:
        cv2.circle(frame, (int(x * 4), int(y * 4)), 5, (0, 255, 0), -1)
    return frame

def main(config_path, video_path, output_path):
    cfg = load_config(config_path)
    model_p, model_d = initialize_models(cfg)
    load_pretrained_weights(model_p, cfg['MODEL']['PRETRAINED'])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    model_p.eval()
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_frame = preprocess_frame(frame, cfg['MODEL']['IMAGE_SIZE'])
            feature_output, keypoint_output = model_p(input_frame)
            if isinstance(keypoint_output, list):
                keypoint_output = keypoint_output[-1]
            keypoints = get_keypoints_from_heatmap(keypoint_output, cfg['MODEL']['IMAGE_SIZE'])

            frame_with_keypoints = draw_keypoints_on_frame(frame, keypoints)
            out.write(frame_with_keypoints)

            # Display the resulting frame
            cv2.imshow('Frame', frame_with_keypoints)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config_path = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
    video_path = 'video.mov'
    output_path = 'output_video_with_joints.mp4'
    main(config_path, video_path, output_path)
