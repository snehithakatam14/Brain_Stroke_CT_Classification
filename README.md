# AI-Based Stroke CT Analysis System

Deep Learning-based Brain CT classification system with Explainable AI using Grad-CAM.

## Features
- Classifies CT scans into:
  - Bleeding
  - Ischemia
  - Normal
- Out-of-distribution rejection mechanism
- Grad-CAM visualization
- Streamlit Web Interface

## Tech Stack
- PyTorch
- ResNet18
- Streamlit
- Plotly
- Grad-CAM

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py



Note: The trained model file (.pth) is not included due to size.
Please add your trained model as best_stroke_model.pth in the project root.



Disclaimer:

For academic purposes only.