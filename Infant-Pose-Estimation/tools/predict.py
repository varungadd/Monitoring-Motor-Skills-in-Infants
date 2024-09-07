import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import _init_paths
from config import cfg
from config import update_config
from utils.utils import create_logger
import numpy as np
import cv2
import models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test keypoints network on a single image"
    )
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
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
    parser.add_argument("--model-file", help="model file", required=True, type=str)
    parser.add_argument(
        "--image", help="path to the image file", required=True, type=str
    )
    parser.add_argument(
        "--output", help="path to save the output", required=True, type=str
    )
    args = parser.parse_args()
    return args


def load_image(image_path, input_size):
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, _, _ = create_logger(cfg, args.cfg, "valid")
    logger.info("Loading model...")
    model_p, _ = eval("models." + cfg.MODEL.NAME + ".get_adaptive_pose_net")(
        cfg, is_train=False
    )
    model_p.load_state_dict(torch.load(args.model_file), strict=False)
    model_p = model_p.cuda()
    model_p.eval()

    image = load_image(args.image, cfg.MODEL.IMAGE_SIZE)
    image = image.cuda()

    with torch.no_grad():
        output = model_p(image)
        print(output[0])
        print(output[0].shape)
        print(output[0].cpu().numpy())
        print(output[1])
        print(output[1].shape)
        print(output[1].cpu().numpy())
        
        # output = output.cpu().numpy()

    np.save(args.output, output)
    print(f"Output saved to {args.output}")

def visualize_pose(image_path, keypoints):
    image = cv2.imread(image_path)
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imshow("Pose Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
