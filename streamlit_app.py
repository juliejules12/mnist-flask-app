import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model
model = tf.keras.models.load_model("weights.h5")

st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = ImageOps.invert(image.resize((28, 28)))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    st.write(f"**Predicted Digit:** {np.argmax(prediction)}")
