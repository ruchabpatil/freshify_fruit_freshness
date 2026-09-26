import json
import base64
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from http.server import BaseHTTPRequestHandler


# --------------------------------------------------
# Load model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "multi_fruit_mlp_baseline.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(image_bytes):

    # Open image and convert to RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize exactly like the Streamlit application
    image = image.resize((32, 32))

    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize
    image_array = image_array / 255.0

    # Flatten to 3072 features
    image_flattened = image_array.reshape(1, 3072)

    # Model prediction
    prediction = model.predict(
        image_flattened,
        verbose=0
    )[0][0]

    prediction = float(prediction)

    # Interpret prediction
    if prediction >= 0.5:
        predicted_class = "Rotten"
    else:
        predicted_class = "Fresh"

    fresh_score = 1.0 - prediction
    rotten_score = prediction

    return {
        "prediction": predicted_class,
        "fresh_score": round(fresh_score * 100, 2),
        "rotten_score": round(rotten_score * 100, 2)
    }


# --------------------------------------------------
# Vercel API Handler
# --------------------------------------------------

class handler(BaseHTTPRequestHandler):

    def do_POST(self):

        try:

            content_length = int(
                self.headers.get("Content-Length", 0)
            )

            body = self.rfile.read(content_length)

            data = json.loads(body)

            image_data = data.get("image")

            if not image_data:
                self.send_response(400)
                self.send_header(
                    "Content-Type",
                    "application/json"
                )
                self.end_headers()

                self.wfile.write(
                    json.dumps({
                        "error": "No image provided."
                    }).encode()
                )

                return

            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)

            result = predict_image(image_bytes)

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json"
            )
            self.end_headers()

            self.wfile.write(
                json.dumps(result).encode()
            )

        except Exception as error:

            self.send_response(500)
            self.send_header(
                "Content-Type",
                "application/json"
            )
            self.end_headers()

            self.wfile.write(
                json.dumps({
                    "error": str(error)
                }).encode()
            )
