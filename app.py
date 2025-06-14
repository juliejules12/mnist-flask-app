from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model("mnist_cnn_model.h5")

def preprocess_image(image):
    image = image.convert("L")  # grayscale
    image = image.resize((28, 28))  # resize
    img_array = np.array(image) / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file)
            processed = preprocess_image(image)
            prediction = np.argmax(model.predict(processed), axis=-1)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
