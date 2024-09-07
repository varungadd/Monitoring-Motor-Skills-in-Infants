import torch
import numpy as np
import cv2
from torchvision import transforms
import yaml
import _init_paths
from core.function import validate_feature
from core.inference import get_final_preds
from models.adaptive_pose_hrnet import get_adaptive_pose_net
from utils.vis import save_batch_image_with_joints

def load_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_p, _ = get_adaptive_pose_net(config, is_train=False)
    model_p.init_weights(config['MODEL']['PRETRAINED'])
    model_p.eval()
    return model_p, config

# Preprocess the image
def preprocess_image(image_path, config):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config['MODEL']['IMAGE_SIZE'][1], config['MODEL']['IMAGE_SIZE'][0]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Visualize the keypoints on the image
def visualize_keypoints(image, keypoints, joints_vis, file_name, config):
    # # Denormalize the image
    image = image.cpu().squeeze(0).numpy()  # Remove batch dimension
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std[:, None, None] + mean[:, None, None]) * 255  # Denormalize and scale to 0-255
    image = image.astype(np.uint8).transpose(1, 2, 0)  # Convert to uint8 and change shape to (H, W, C)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
    image2 = cv2.imread('image3.jpg')
    image2 = cv2.resize(image, (config['MODEL']['IMAGE_SIZE'][1], config['MODEL']['IMAGE_SIZE'][0]))
    # Save intermediate image for debugging
    cv2.imwrite('temp.jpg', image)

    for joint, joint_vis in zip(keypoints, joints_vis):
        if joint_vis[0]:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 2, (255, 0, 0), 2)
            cv2.circle(image2, (int(joint[0]), int(joint[1])), 2, (255, 0, 0), 2)
    
    cv2.imwrite(file_name, image)
    cv2.imwrite("output_pred2.jpg", image2)

# Make prediction on a single image
def predict_on_single_image(image_path, model, config):
    image = preprocess_image(image_path, config)
    with torch.no_grad():
        feature_outputs, outputs = model(image)
        # print(outputs)
        output = outputs[-1] if isinstance(outputs, list) else outputs
        
        c = np.array([[image.shape[3] // 2, image.shape[2] // 2]], dtype=np.float32)
        s = np.array([[1.0, 1.0]], dtype=np.float32)
        
        preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
        
        joints_vis = np.ones((preds.shape[1], 1), dtype=np.float32)  # Assuming all joints are visible
        print(image.shape, preds.shape, joints_vis.shape)
        print(image.size(0))
        save_batch_image_with_joints(batch_image=image, batch_joints=preds*4, batch_joints_vis=joints_vis, file_name='out2.jpg')
        # visualize_keypoints(image[0], preds[0]*4, joints_vis, 'output_pred.jpg', config)

# Main function to run the prediction
def main():
    config_file = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'  # Update this path
    image_path = 'image3.jpg'
    model, cfg = load_model(config_file)
        
    predict_on_single_image(image_path, model, cfg)

if __name__ == '__main__':
    main()
