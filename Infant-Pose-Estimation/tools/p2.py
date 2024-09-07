import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import _init_paths
from config import cfg, update_config
import models

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test keypoints network on a single image"
    )
    parser.add_argument(
        "--cfg", help="experiment configure file name", type=str, default="experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument("--model-file", help="model file",  type=str)
    parser.add_argument(
        "--image", help="path to the image file", type=str
    )
    parser.add_argument(
        "--output", help="path to save the output", type=str
    )
    args = parser.parse_args()
    return args

def load_model(cfg_path, model_path):
    args = parse_args()
    update_config(cfg, args)
    # update_config(cfg, cfg_path)
    
    model_p, model_d = eval('models.' + cfg.MODEL.NAME + '.get_adaptive_pose_net')(
        cfg, is_train=False
    )
    
    model_p.load_state_dict(torch.load(model_path), strict=False)
    model_p = torch.nn.DataParallel(model_p, device_ids=cfg.GPUS).cuda()
    model_p.eval()
    
    return model_p

def preprocess_image(image_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    image = Image.open(image_path).convert('RGB')
    image = image.resize((384, 288))  # Resize to match model input size
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_pose(model, image):
    with torch.no_grad():
        outputs = model(image)
    return outputs

def visualize_pose(image_path, keypoints):
    image = cv2.imread(image_path)
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imshow("Pose Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # args = parse_args()
    # update_config(cfg, args)
    cfg_path = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
    model_path = 'models/hrnet_fidip.pth'
    image_path = '../image.png'
    
    model = load_model(cfg_path, model_path)
    image = preprocess_image(image_path)
    keypoints = predict_pose(model, image)
    
    # Assuming keypoints is a numpy array of shape (17, 2) representing (x, y) coordinates
    visualize_pose(image_path, keypoints.cpu().numpy().reshape(-1, 2))
