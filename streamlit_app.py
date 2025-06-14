import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 image of a digit (0-9) in grayscale.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors if necessary

    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess image
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.markdown(f"### 🔢 Predicted Digit: `{predicted_digit}`")
