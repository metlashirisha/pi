import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Debug: Print model input shape
st.write(f"✅ Model Input Shape: {model.input_shape}")

# Extract input shape
if isinstance(model.input_shape, list):
    input_shape = model.input_shape[0]  # Handle list case
else:
    input_shape = model.input_shape

if input_shape is not None and len(input_shape) == 4:
    _, img_height, img_width, img_channels = input_shape  # Ignore batch dimension
else:
    st.error("❌ Invalid model input shape. Please check the model architecture.")
    st.stop()

# Define image preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize((img_width, img_height))  # Resize to model's expected size
    image = np.array(image) / 255.0  # Normalize pixel values

    # Ensure correct number of channels
    if image.shape[-1] != img_channels:
        st.error(f"❌ Expected {img_channels} channels, but got {image.shape[-1]}")
        st.stop()

    image = np.expand_dims(image, axis=0)  # Add batch dimension
    st.write(f"✅ Processed Image Shape: {image.shape}")  # Debug output
    return image.astype(np.float32)

# Streamlit UI
st.title("🧠 Autism Facial Recognition Model")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    try:
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)

        # Display result
        st.write(f"🔍 Prediction: {prediction}")
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
