import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Custom Grad-CAM for YOLOv11
class YOLOv11GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()

        output = self.model.model(input_tensor)[0]
        target = output[..., 4].sum()
        target.backward()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam -= cam.min()
        cam /= cam.max()
        return cam

# --- Setup ---
name = "yolo11_4head_4ca_carafe"  # or "yolo11l"
number = 13 if name == "yolo11_4head_4ca_carafe" else 9

# Load model
model = YOLO(f"runs/detect/{name}/weights/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device).eval()
target_layer = model.model.model[number]
cam_extractor = YOLOv11GradCAM(model, target_layer)

# --- Image paths ---
img_dir = "dataset_canada_7_3/images/val"
output_dir = f"runs/detect/heatmaps/{name}"
os.makedirs(output_dir, exist_ok=True)

# --- Process all images ---
for fname in os.listdir(img_dir):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
    img_path = os.path.join(img_dir, fname)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb_norm = image_rgb / 255.0

    img_resized = cv2.resize(image_rgb_norm, (640, 640))
    img_input = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    img_input.requires_grad = True

    cam = cam_extractor.generate_cam(img_input)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + cv2.resize(image, (640, 640)) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, overlay)

print(f"✅ Saved all heatmaps to {output_dir}")
