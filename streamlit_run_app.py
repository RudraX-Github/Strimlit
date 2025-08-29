import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json
import urllib.request
import io

# Try importing timm with error handling
try:
    import timm
except ModuleNotFoundError:
    st.error("❌ The 'timm' library is not installed. Please run `pip install timm` in your terminal.")
    st.stop()

# Title and description
st.title("ConvNeXt Image Classifier 🌿")
st.write("Choose input method and model type, then click 'Predict' to classify your image.")

# Model selector
model_choice = st.radio("🧠 Choose model", ["ImageNet-1k", "ImageNet-22k"])

# Input method selector
input_method = st.selectbox("📌 Select input method", ["Upload from device", "Capture with camera"])

# Image input based on selection
image = None
if input_method == "Upload from device":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif input_method == "Capture with camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(io.BytesIO(camera_image.getvalue())).convert("RGB")

# Predict button
predict_button = st.button("🔍 Predict")

# Placeholder for result
result_placeholder = st.empty()

# Cache model loading
@st.cache_resource
def load_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    return model

# Cache label map loading
@st.cache_resource
def load_label_map(model_name):
    if "in1k" in model_name:
        url = "https://huggingface.co/datasets/imagenet-1k/resolve/main/labels.json"
    else:
        url = "https://huggingface.co/datasets/huggingface/label-files/resolve/main/imagenet-22k-id2label.json"
    try:
        with urllib.request.urlopen(url) as response:
            return json.load(response)
    except Exception:
        return {}

# Display image and run prediction
if image:
    st.image(image, caption="Selected Image", use_container_width=True)

    if predict_button:
        with st.spinner("Classifying..."):
            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_tensor = transform(image).unsqueeze(0)

            # Choose model name
            model_name = "convnext_large.in1k" if model_choice == "ImageNet-1k" else "convnext_large.fb_in22k"

            # Load model and labels
            model = load_model(model_name)
            label_map = load_label_map(model_name)

            # Inference
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)[0]
                top5 = torch.topk(probs, k=5)

            # Display top-5 predictions
            st.subheader("🔝 Top 5 Predictions")
            for i in range(5):
                idx = top5.indices[i].item()
                score = top5.values[i].item()
                label = label_map.get(str(idx), f"Class {idx}")
                st.write(f"**{i+1}. {label}** — Confidence: `{score:.2%}`")

            # Show most confident prediction
            result_placeholder.success(f"✅ Most Likely: {label_map.get(str(top5.indices[0].item()), 'Unknown')}")
