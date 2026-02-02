import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras

APP_HOST = "0.0.0.0"
APP_PORT = int(os.environ.get("PORT", "8080"))

MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
LABELS = ["FR0", "FRI", "FRII"]  # index -> label (locked)

# Load once at startup (fast per-request inference)
model = keras.models.load_model(MODEL_PATH, compile=False)

app = Flask(__name__)

def preprocess_image_bytes(img_bytes: bytes) -> np.ndarray:
    # Decode (supports JPEG/PNG); returns uint8 tensor [H,W,C]
    x = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)
    # Resize to model input
    x = tf.image.resize(x, (60, 60), method="bilinear")
    # Convert to float32 and normalize /255
    x = tf.cast(x, tf.float32) / 255.0
    # Add batch dim -> [1,60,60,1]
    x = tf.expand_dims(x, axis=0)
    return x.numpy()

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.post("/")
def predict():
    ct = request.content_type or ""
    if ct not in ("image/jpeg", "image/png"):
        return jsonify(error="Unsupported Content-Type. Use image/jpeg or image/png."), 415

    img_bytes = request.get_data(cache=False)
    if not img_bytes:
        return jsonify(error="Empty request body. Send raw image bytes."), 400

    x = preprocess_image_bytes(img_bytes)

    # Inference
    probs = model.predict(x, verbose=0)[0].astype(float)  # shape (3,)
    idx = int(np.argmax(probs))
    out = {
        "predicted_index": idx,
        "predicted_label": LABELS[idx],
        "probabilities": probs.tolist(),
    }
    return jsonify(out)

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)
