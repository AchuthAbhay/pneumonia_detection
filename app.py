import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import models
import torch.nn as nn

model = models.resnet50(pretrained=False)  # Use ResNet-50, not resnet18 or others
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes for pneumonia detection

model.load_state_dict(torch.load('best_resnet_model.pth', map_location='cpu'))
model.eval()

# Grad-CAM hook
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        output = self.model(input_tensor)
        class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        gradients = self.gradients.squeeze(0)
        activations = self.activations.squeeze(0)

        weights = gradients.mean(dim=(1, 2))
        weights = weights.to(activations.device)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        return cam.cpu().numpy(), class_idx

# Grad-CAM Setup
grad_cam = GradCAM(model, model.layer4[-1])

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction + Grad-CAM + Overlay
def predict_with_heatmap(img_pil):
    img_tensor = transform(img_pil).unsqueeze(0)
    mask, class_idx = grad_cam.generate_cam(img_tensor)

    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    mask_resized = cv2.resize(mask, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = heatmap * 0.5 + img_np * 0.5
    overlay = overlay / overlay.max()
    return overlay, class_idx

# Streamlit UI
st.title("🩺 Pneumonia Detection with Grad-CAM")
st.write("Upload a chest X-ray to detect Pneumonia and visualize the model's decision with Grad-CAM.")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.write("Generating prediction and heatmap...")
    overlay_img, class_idx = predict_with_heatmap(img)
    label = "PNEUMONIA" if class_idx == 1 else "NORMAL"
    st.success(f"Prediction: {label}")

    st.image(overlay_img, caption="Grad-CAM Heatmap", use_container_width=True)

    st.markdown("""
    ### What does the heatmap tell us?
    - **Bright/red/yellow regions** highlight areas in the chest X-ray that influenced the model's prediction the most.
    - **Cooler regions** indicate less important areas for the decision.
    - This helps us understand *why* the model predicted pneumonia or normal by showing the focus regions.
    - Useful for building trust and verifying that the model looks at relevant lung areas.
    """)
