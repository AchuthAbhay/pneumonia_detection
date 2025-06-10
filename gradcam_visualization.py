import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 1. Hook to capture gradients of the target convolutional layer
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Register hooks on target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
      output = self.model(input_tensor)
      class_idx = torch.argmax(output, dim=1).item()

      self.model.zero_grad()
      class_score = output[0, class_idx]
      class_score.backward()

      gradients = self.gradients.squeeze(0).detach()
      activations = self.activations.squeeze(0).detach()

      weights = gradients.mean(dim=(1, 2))  # Global Average Pooling
      weights = weights.to(activations.device)  # 🔧 Fix: Move weights to same device

      cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)  # 🔧 Match device
      for i, w in enumerate(weights):
        cam += w * activations[i]

      cam = F.relu(cam)
      cam = cam - cam.min()
      cam = cam / cam.max()
      return cam.cpu().numpy()

# 2. Preprocessing for input image (similar to your training transforms)
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                             std=[0.229, 0.224, 0.225])
    ])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = transforms.ToPILImage()(img)
    tensor = transform(pil_img).unsqueeze(0)  # Add batch dim
    return tensor, img

def show_cam_on_image(img, mask, alpha=0.5):
    # Resize mask to match original image size
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Normalize mask
    mask = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Ensure both images are float32 and scaled between 0 and 1
    heatmap = np.float32(heatmap) / 255.0
    img = np.float32(img) / 255.0

    # Blend heatmap and original image
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = overlay / overlay.max()

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Grad-CAM Overlay')
    plt.show()


from torchvision import models
import torch.nn as nn

# Load the base pretrained model
model = models.resnet18(pretrained=True)

# Modify the final layer to match your number of classes
num_classes = 2  # NORMAL and PNEUMONIA
model.fc = nn.Linear(model.fc.in_features, num_classes)


# 4. Usage example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your trained model (make sure model is in eval mode)
model.eval()
model.to(device)

# Replace 'layer4' with the last convolutional layer in your ResNet variant
target_layer = model.layer4[-1].conv2

grad_cam = GradCAM(model, target_layer)

# Image path from your dataset (replace with your test image path)
img_path = 'data/chest_xray/test/NORMAL/IM-0001-0001.jpeg'

input_tensor, original_img = preprocess_image(img_path)
input_tensor = input_tensor.to(device)

# Generate Grad-CAM mask
mask = grad_cam.generate_cam(input_tensor)

# Show heatmap overlay on original image
show_cam_on_image(original_img, mask)
