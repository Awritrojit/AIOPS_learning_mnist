import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import mlflow.pytorch
import os

# Get the absolute path to the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Load the trained PyTorch model from MLflow
model_path = os.path.join(current_dir, "mlruns/0/c89566f24f7b4bf0b526c7ac29c426be/artifacts/model")
model = mlflow.pytorch.load_model(model_path)
model.eval()

# Image transformations
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Streamlit app
st.title("MNIST Digit Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image and make a prediction
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img_tensor)

    # Get the predicted digit
    prediction = torch.argmax(F.softmax(output[0], dim=0)).item()
    st.write(f"Prediction: {prediction}")
