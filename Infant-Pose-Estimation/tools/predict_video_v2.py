import torch
import numpy as np
import cv2
from torchvision import transforms
import yaml
import _init_paths
from core.function import validate_feature
from core.inference import get_final_preds, get_max_preds
from models.adaptive_pose_hrnet import get_adaptive_pose_net
from utils.vis import save_batch_image_with_joints
import argparse

def calculate_center_scale(image, config):
    height, width = config['MODEL']['IMAGE_SIZE']
    center = np.array([width // 2, height // 2], dtype=np.float32)

    aspect_ratio = width / height
    pixel_std = 200

    if width > aspect_ratio * height:
        height = width / aspect_ratio
    else:
        width = height * aspect_ratio

    scale = np.array([width / pixel_std, height / pixel_std], dtype=np.float32)
    return center, scale

def load_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_p, _ = get_adaptive_pose_net(config, is_train=False)
    model_p.load_state_dict(torch.load('models/hrnet_fidip.pth'), strict=False)
    return model_p, config

def preprocess_frame(frame, config):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (config['MODEL']['IMAGE_SIZE'][0], config['MODEL']['IMAGE_SIZE'][1]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_transformed = transform(frame_resized).unsqueeze(0)  # Add batch dimension
    return frame_transformed, frame_rgb.shape[0:2]

def visualize_keypoints(frame, keypoints, joints_vis, original_shape):
    # Denormalize the frame
    frame = frame.cpu().squeeze(0).numpy()  # Remove batch dimension
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame * std[:, None, None] + mean[:, None, None]) * 255  # Denormalize and scale to 0-255
    frame = frame.astype(np.uint8).transpose(1, 2, 0)  # Convert to uint8 and change shape to (H, W, C)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV

    for joint, joint_vis in zip(keypoints, joints_vis):
        if joint_vis[0]:
            cv2.circle(frame, (int(joint[0]), int(joint[1])), 2, (255, 0, 0), 2)

    # Resize back to the original shape
    frame_resized = cv2.resize(frame, original_shape[::-1])
    return frame_resized

def predict_on_frame(frame, model, config):
    frame_transformed, original_shape = preprocess_frame(frame, config)
    with torch.no_grad():
        feature_outputs, outputs = model(frame_transformed)

        output = outputs[-1] if isinstance(outputs, list) else outputs
        pred, _ = get_max_preds(output.clone().cpu().numpy())
        c, s = calculate_center_scale(frame_transformed, config)
        preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
        joints_vis = np.ones((1, preds.shape[1], 1), dtype=np.float32)  # Assuming all joints are visible
        preds = np.concatenate((preds, maxvals), axis=-1)  # Shape: (1, 17, 3)
        preds = torch.from_numpy(preds).float()  # Convert to torch tensor

        frame_with_keypoints = visualize_keypoints(frame_transformed[0], pred[0][:, :2] * 4, joints_vis[0], original_shape)
        return frame_with_keypoints

def process_video(video_path, output_path, model, config):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_with_keypoints = predict_on_frame(frame, model, config)
        out.write(frame_with_keypoints)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main(video_path):
    config_file = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'  # Update this path
    output_path = 'output_video.mp4'
    model, cfg = load_model(config_file)
    process_video(video_path, output_path, model, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to the video file')
    args = parser.parse_args()
    main(args.video)
