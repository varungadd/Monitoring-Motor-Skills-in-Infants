import torch
import cv2
import yaml
import numpy as np
from torchvision import transforms
import _init_paths
from models.adaptive_pose_hrnet import get_adaptive_pose_net


def load_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_p, _ = get_adaptive_pose_net(config, is_train=False)
    model_p.init_weights(config['MODEL']['PRETRAINED'])
    model_p.eval()
    return model_p, config

config_path = 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
model, config = load_model(config_path)

print("Model and configuration loaded successfully!")

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

image_path = 'image2.jpg'
input_size = config['MODEL']['IMAGE_SIZE']
input_image = preprocess_image(image_path, input_size)

with torch.no_grad():
    feature_output, keypoints_output = model(input_image)
    print("Feature output shape:", feature_output.shape)
    print("Keypoints output shape:", keypoints_output.shape)

# keypoints_output = keypoints_output.squeeze().cpu().numpy()
# print(keypoints_output)
print(keypoints_output.shape)

def get_max_preds(heatmaps):
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 4, 'heatmaps should be 4-ndim'

    N, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((N, K, 1))
    idx = idx.reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % W
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / W)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

keypoints_heatmap = keypoints_output.cpu().numpy()
preds, maxvals = get_max_preds(keypoints_heatmap)
keypoints = preds[0]  # Taking the keypoints for the first (and only) image


def visualize_keypoints(image_path, keypoints, output_path, input_size):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Scale keypoints to the original image size
    keypoints[:, 0] *= (w / input_size[1])
    keypoints[:, 1] *= (h / input_size[0])

    for joint in keypoints:
        if joint[0] >= 0 and joint[1] >= 0:  # Only draw valid keypoints
            cv2.circle(image, (int(joint[0]), int(joint[1])), 9, [255, 0, 0], 2)
    cv2.imwrite(output_path, image)


output_path = 'annotated_image.jpg'
visualize_keypoints(image_path, keypoints, output_path, input_size)



# keypoints = keypoints_output[0].cpu().numpy()  # Assuming the output is of shape [1, num_joints, 3]
# keypoints = keypoints[:, :2]  # Extracting x, y coordinates
# output_path = 'annotated_image.jpg'
# visualize_keypoints(image_path, keypoints, output_path)
