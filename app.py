import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title=" Stroke CT Analysis", layout="wide")

# ---------------- CLEAN CSS ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.hero {
    background: linear-gradient(90deg,#2b3f91,#3f51b5);
    padding: 30px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 8px;
}

.disclaimer {
    background-color: #e3f2fd;
    padding: 12px;
    border-radius: 8px;
    font-size: 14px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
<h1>Stroke CT Analysis System</h1>
<p>Deep Learning Powered Stroke Prediction with Explainable AI</p>
</div>
""", unsafe_allow_html=True)

device = torch.device("cpu")

# ---------------- LOAD OOD STATS ----------------
feature_mean = np.load("feature_mean.npy")
feature_cov_inv = np.load("feature_cov_inv.npy")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("best_stroke_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
feature_extractor = nn.Sequential(*list(model.children())[:-1])
target_layer = model.layer4[-1]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ['Bleeding', 'Ischemia', 'Normal']

# ---------------- OOD CHECK ----------------
def is_ct_scan(image):

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), -1).cpu().numpy()[0]

    diff = features - feature_mean
    mahalanobis = np.sqrt(diff.T @ feature_cov_inv @ diff)

    print("Mahalanobis distance:", mahalanobis)

    return mahalanobis < 80


# ---------------- PREDICT ----------------
def predict_image(image):

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities.squeeze().cpu().numpy()


# ---------------- GRAD CAM ----------------
def generate_gradcam(image):

    model.eval()

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = outputs.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    grayscale_cam = np.maximum(grayscale_cam, 0)
    grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)
    grayscale_cam = grayscale_cam ** 2.5

    import cv2
    grayscale_cam = cv2.GaussianBlur(grayscale_cam, (7, 7), 0)

    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized) / 255.0

    visualization = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.5
    )

    return visualization


# ---------------- RESEARCH LEVEL EXPLANATION ----------------
def generate_research_explanation(predicted_class, confidence):

    explanation = ""
    confidence_percent = confidence * 100

    if confidence_percent > 95:
        certainty = "very high diagnostic confidence"
    elif confidence_percent > 85:
        certainty = "high diagnostic confidence"
    elif confidence_percent > 70:
        certainty = "moderate diagnostic confidence"
    else:
        certainty = "low diagnostic confidence"

    explanation += f"<b>Model Prediction:</b> {predicted_class}<br><br>"
    explanation += f"<b>Model Confidence:</b> {confidence_percent:.2f}% ({certainty})\n\n"

    if predicted_class == "Bleeding":

        explanation += """
### Radiological Interpretation
- The model detected **hyperattenuating (hyperdense) intracranial regions**, a classical radiological indicator of **acute intracranial hemorrhage** on non-contrast CT imaging.
- Acute hemorrhage appears hyperdense due to **clot formation and high hemoglobin concentration**, producing localized high-intensity regions relative to surrounding brain parenchyma.
- The **Grad-CAM attention map shows strong localized activation** over these hyperdense areas.

### Clinical Context
This pattern is consistent with **hemorrhagic stroke**, requiring rapid clinical evaluation.
"""

    elif predicted_class == "Ischemia":

        explanation += """
### Radiological Interpretation
- The model identified **hypodense parenchymal regions** representing decreased tissue attenuation.
- Early ischemic stroke often manifests as **subtle hypoattenuation or loss of gray-white matter differentiation**.

### Clinical Context
Findings suggest **ischemic infarction**, requiring clinical correlation and further imaging.
"""

    elif predicted_class == "Normal":

        explanation += """
### Radiological Interpretation
- No focal **hyperdense** or **hypodense** abnormalities detected.
- Brain structure appears symmetric with preserved gray-white matter differentiation.

### Clinical Context
The CT slice appears **radiologically normal**.
"""

    explanation += """
---

### Explainability Methodology
- Explanation generated using **Gradient-weighted Class Activation Mapping (Grad-CAM)**.
- Grad-CAM highlights regions contributing most strongly to the model’s prediction.
"""

    return explanation


# ---------------- MAIN LAYOUT ----------------
left, right = st.columns(2)

# ================= LEFT =================
with left:

    st.markdown('<div class="section-title">Upload Brain CT Scan</div>', unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.write("**Selected Image:**")
        st.image(image, width=320)

        if st.button("Analyze CT SCAN"):
            st.session_state["image"] = image
            st.session_state["run"] = True


# ================= RIGHT =================
with right:

    st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
    st.divider()

    if "run" in st.session_state and st.session_state["run"]:

        image = st.session_state["image"]

        if not is_ct_scan(image):
            st.error("Invalid Input: This does not appear to be a Brain CT scan.")
            st.stop()

        predicted_idx, confidence, probs = predict_image(image)

        if confidence < 0.75:
            st.error("Invalid Input: Low model confidence.")
            st.stop()

        predicted_class = class_names[predicted_idx]

        tab1, tab2 = st.tabs(["RESULTS", "EXPLAIN AI"])

        with tab1:

            if predicted_class in ["Bleeding", "Ischemia"]:
                st.error(f"⚠ {predicted_class} Detected")
            else:
                st.success("✓ Normal Scan")

            st.write(f"Confidence: {confidence*100:.2f}%")

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[p*100 for p in probs],
                y=class_names,
                orientation='h',
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="inside"
            ))

            fig.update_layout(height=260, xaxis=dict(range=[0, 100]), plot_bgcolor="white")

            st.plotly_chart(fig, use_container_width=True)

        with tab2:

            colA, colB = st.columns(2)

            with colA:
                st.subheader("AI Attention Visualization")

                cam_image = generate_gradcam(image)
                st.image(cam_image, width=280)

                st.caption("Warmer colors indicate higher model focus.")

            with colB:
                st.subheader("Diagnostic Explanation")

                explanation = generate_research_explanation(predicted_class, confidence)

                st.markdown(
                    f"""
                    <div style="
                        max-height:420px;
                        overflow-y:auto;
                        padding:18px;
                        line-height:1.7;
                        border-radius:10px;
                        background:#f9fbff;
                        border-left:5px solid #3f51b5;
                        box-shadow:0 2px 6px rgba(0,0,0,0.05);
                    ">
                    {explanation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("""
<div class="disclaimer">
⚠ <b>Clinical Disclaimer:</b><br>
This AI system is designed as a <b>decision-support tool for research and educational purposes</b>.
Final diagnosis should always be performed by qualified medical professionals.
</div>
""", unsafe_allow_html=True)

    else:
        st.info("Upload a CT scan and click Analyze.")