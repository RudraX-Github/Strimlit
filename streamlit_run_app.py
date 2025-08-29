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
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.warning(f"⚠️ Failed to load uploaded image: {e}")
elif input_method == "Capture with camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        try:
            image = Image.open(io.BytesIO(camera_image.getvalue())).convert("RGB")
        except Exception as e:
            st.warning(f"⚠️ Failed to process camera image: {e}")

# Predict button
predict_button = st.button("🔍 Predict")

# Placeholder for result
result_placeholder = st.empty()

# Cache model loading using torch.load
@st.cache_resource
def load_model(model_name):
    model = timm.create_model(model_name, pretrained=False)
    checkpoint_path = "convnext_large_in1k.pth" if "in1k" in model_name else "convnext_large_fb_in22k.pth"
    try:
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model checkpoint: {e}")
        st.stop()

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
if image is not None:
    try:
        st.image(image, caption="Selected Image", use_column_width=True)
    except TypeError as e:
        st.warning(f"⚠️ Image display failed: {e}")

    st.write(f"🧪 Debug: Image type is `{type(image)}`")

    if predict_button:
        with st.spinner("Classifying..."):
            try:
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
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
else:
    st.warning("⚠️ No image selected or failed to load. Please upload or capture an image to proceed.")
