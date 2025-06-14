import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load your trained MNIST model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("🖌️ Draw a Digit - MNIST Classifier")
st.write("Draw a digit (0–9) below and I’ll try to guess it!")

# Canvas component for drawing
canvas_result = st_canvas(
    fill_color="black",  # Background fill color
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Extract the drawing from canvas
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))  # Use red channel
    img = img.resize((28, 28)).convert("L")  # Resize and convert to grayscale

    # Show model input preview
    st.image(img.resize((140, 140)), caption="🧐 Model Input Preview")

    # Preprocess for model
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.success(f"✅ I think it's a **{predicted_digit}**.")
