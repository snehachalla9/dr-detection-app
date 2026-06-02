import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import joblib
from PIL import Image
import numpy as np
import torchvision.transforms as T
DEVICE = torch.device("cpu")

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

# ---------------- Model ----------------
class EfficientNetB0_DR(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4):
        super().__init__()

        base = models.efficientnet_b0(weights=None)

        self.features = base.features
        self.avgpool = base.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


@st.cache_resource
def load_models():
    model = EfficientNetB0_DR().to(DEVICE)

    #model = EfficientNetB0_DR()

    model.load_state_dict(
        torch.load(
            "effb0_se_best.pth",
            map_location="cpu"
        )
    )

    model.eval()

    svm = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, svm, scaler
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]
DESCRIPTIONS = {
    "No DR": "No signs of diabetic retinopathy detected.",
    "Mild": "Early retinal abnormalities detected.",
    "Moderate": "Moderate retinal damage detected.",
    "Severe": "Significant retinal damage detected.",
    "Proliferative": "Advanced diabetic retinopathy detected. Immediate ophthalmologist consultation is recommended."
}
def predict_image(image):

    # image = transform(image).unsqueeze(0)
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.extract_features(image)
        features = features.cpu().numpy()

    features = scaler.transform(features)

    prediction = svm.predict(features)[0]

    return CLASS_NAMES[int(prediction)]


# ---------------- Load Models ----------------
try:
    model, svm, scaler = load_models()

except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ---------------- UI ----------------
# st.title("👁️ Diabetic Retinopathy Detection")
st.markdown("""
<h1 style='text-align:center; color:#1E88E5;'>
👁️ Diabetic Retinopathy Detection
</h1>
<h4 style='text-align:center;'>
AI-Powered Retinal Screening System
</h4>
""", unsafe_allow_html=True)

st.divider()

st.markdown(
    """
    Upload a retinal fundus image to predict the stage of
    diabetic retinopathy using our deep learning model.
    """
)

# uploaded_file = st.file_uploader(
#     "Upload Retinal Image",
#     type=["jpg", "jpeg", "png"]
# )
st.subheader("📤 Upload Retinal Fundus Image")

uploaded_file = st.file_uploader(
    "",
    type=["jpg","jpeg","png"]
)

if uploaded_file:
    with st.container(border=True):
        st.image(
        uploaded_file,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    st.markdown("### Ready for Analysis")
    if st.button("🔍 Predict DR Stage", width="stretch"):
        image = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Analyzing retinal image..."):
            prediction = predict_image(image)
            STATUS = {
                "No DR":"🟢",
                "Mild":"🟡",
                "Moderate":"🟠",
                "Severe":"🔴",
                "Proliferative":"🚨"
            }
            st.success(
                f"{STATUS[prediction]} {prediction}"
                )
            st.info(DESCRIPTIONS[prediction])
    # if st.button("🔍 Predict DR Stage", use_container_width=True):
    #     image = Image.open(uploaded_file).convert("RGB")
    #     prediction = predict_image(image)
    #     st.success(f"Prediction: {prediction}")

    # if st.button("🔍 Predict DR Stage", use_container_width=True):
    #     st.info("Prediction pipeline will be added next.")
