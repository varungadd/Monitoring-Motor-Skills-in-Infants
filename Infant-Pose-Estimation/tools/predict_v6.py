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
    # final_output_dir = os.path.join(config['OUTPUT_DIR'], config['MODEL']['MODEL_NAME'])
    final_output_dir = '/home/harsh/Documents/Documents/ml-projects/Monitoring-Motar-Skills-in-Infants/Infant-Pose-Estimation/output'
    model_p.load_state_dict(torch.load('models/hrnet_fidip.pth'), strict=False)
    # model_p.init_weights(config['MODEL']['PRETRAINED'])
    # model_p = torch.nn.DataParallel(model_p, device_ids=config['GPUS']).cuda()
    # model_p.eval()
    return model_p, config

# Preprocess the image
def preprocess_image(image_path, config):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config['MODEL']['IMAGE_SIZE'][0], config['MODEL']['IMAGE_SIZE'][1]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Visualize the keypoints on the image
def visualize_keypoints(image, keypoints, joints_vis, file_name, config):
    # Denormalize the image
    image = image.cpu().squeeze(0).numpy()  # Remove batch dimension
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std[:, None, None] + mean[:, None, None]) * 255  # Denormalize and scale to 0-255
    image = image.astype(np.uint8).transpose(1, 2, 0)  # Convert to uint8 and change shape to (H, W, C)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV

    for joint, joint_vis in zip(keypoints, joints_vis):
        if joint_vis[0]:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 2, (255, 0, 0), 2)
    
    cv2.imwrite(file_name, image)

# Make prediction on a single image
def predict_on_single_image(image_path, model, config):
    image = preprocess_image(image_path, config)
    print("Image shape:", image.shape)
    with torch.no_grad():
        # read from input.npy file
        # image = torch.from_numpy(np.load('input.npy')).float()
        # image = image.unsqueeze(0)
        feature_outputs, outputs = model(image)

        output = outputs[-1] if isinstance(outputs, list) else outputs
        
        pred, _ = get_max_preds(output.clone().cpu().numpy())
        c, s = calculate_center_scale(image, config)
        # c = np.array([[379.1817, 445.8421]])
        # s = np.array([[4.263158, 5.684211]])
        preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
        joints_vis = np.ones((1, preds.shape[1], 1), dtype=np.float32)  # Assuming all joints are visible

        # Convert preds to the correct shape
        preds = np.concatenate((preds, maxvals), axis=-1)  # Shape: (1, 17, 3)
        preds = torch.from_numpy(preds).float()  # Convert to torch tensor
        
        # save_batch_image_with_joints(batch_image=image, batch_joints=pred*4, batch_joints_vis=joints_vis, file_name='out2.jpg')
        visualize_keypoints(image[0], pred[0][:, :2]*4, joints_vis[0], 'output_pred.jpg', config)

# Main function to run the prediction
def main():
    config_file = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'  # Update this path
    image_path = 'image.png'
    model, cfg = load_model(config_file)
        
    predict_on_single_image(image_path, model, cfg)

if __name__ == '__main__':
    main()
