"""
Bone Fracture Detection — Flask + Grad-CAM
==========================================
Matches the original Streamlit app exactly:
  • Grayscale input  (BGR → GRAY)
  • IMG_SIZE = 128   → shape (1, 128, 128, 1)
  • Sigmoid output   → prediction > 0.5 = Fractured
  • Model auto-downloaded from HuggingFace if not present

Run:  python app.py
Open: http://localhost:5000
"""

import os
import base64
import requests
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

IMG_SIZE   = 128          # must match training (your original uses 128, not 224)
MODEL_PATH = "bone_fracture_cnn_model.h5"
MODEL_URL  = (
    "https://huggingface.co/Sricharan08/bone_fracture_detection"
    "/resolve/main/bone_fracture_cnn_model.h5"
)

# ── Model globals ─────────────────────────────────────────────────────────────
model      = None
grad_model = None   # sub-model: input → [last_conv_output, sigmoid_logit]


# ── Download + load ───────────────────────────────────────────────────────────

def download_model():
    """Download the .h5 from HuggingFace if it isn't already on disk."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"[INFO] Downloading model from HuggingFace…")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"[INFO] Model saved to {MODEL_PATH}")


def find_last_conv_layer(keras_model) -> str:
    """Return the name of the deepest Conv2D layer — target for Grad-CAM."""
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise RuntimeError("No Conv2D layer found — Grad-CAM requires a CNN.")


def build_grad_model(keras_model, conv_name: str):
    """Sub-model that outputs [conv_feature_maps, final_sigmoid_output]."""
    return tf.keras.Model(
        inputs=keras_model.inputs,
        outputs=[
            keras_model.get_layer(conv_name).output,
            keras_model.output
        ]
    )


def load_model():
    global model, grad_model
    try:
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[INFO] Model loaded: {MODEL_PATH}")

        conv_name  = find_last_conv_layer(model)
        print(f"[INFO] Grad-CAM target layer: '{conv_name}'")

        grad_model = build_grad_model(model, conv_name)
        print("[INFO] Grad-CAM sub-model ready.")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        model = grad_model = None


# ── Image helpers ─────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_bgr(image_bytes: bytes) -> np.ndarray:
    """Raw bytes → BGR uint8 ndarray (original resolution)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    return img   # BGR, uint8


def preprocess_for_model(bgr_img: np.ndarray) -> np.ndarray:
    """
    Exactly mirrors your original Streamlit preprocess_image():
      BGR → GRAY → resize to 128×128 → /255 → reshape (1, 128, 128, 1)
    """
    gray    = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)          # (H, W)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))              # (128, 128)
    norm    = resized.astype("float32") / 255.0                   # [0, 1]
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)                 # (1,128,128,1)


def to_b64_jpeg(rgb_img: np.ndarray, quality: int = 92) -> str:
    """Encode an RGB uint8 ndarray as a JPEG base64 data-URI."""
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed.")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def bytes_to_b64(image_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(img_array: np.ndarray, is_fractured: bool) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a grayscale (1-channel) input.

    img_array shape: (1, 128, 128, 1)

    Steps:
      1. Forward pass → conv feature maps + sigmoid score.
      2. Differentiate score w.r.t. conv feature maps.
      3. Global-average-pool gradients → per-channel weights.
      4. Weighted sum of maps → ReLU → normalise to [0, 1].

    Returns float32 array (h, w) in [0, 1]  where h, w are the
    spatial dims of the last conv layer (e.g. 14×14 for a 128-input CNN).
    """
    with tf.GradientTape() as tape:
        img_t = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_t)
        tape.watch(conv_outputs)

        # Sigmoid binary: single output neuron
        # Score for the class we want to explain
        score = predictions[:, 0] if is_fractured else (1.0 - predictions[:, 0])

    grads   = tape.gradient(score, conv_outputs)          # (1, h, w, C)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))       # (C,)

    maps    = conv_outputs[0].numpy()                     # (h, w, C)
    weights = weights.numpy()                             # (C,)

    cam = np.zeros(maps.shape[:2], dtype="float32")
    for i, w in enumerate(weights):
        cam += w * maps[:, :, i]

    cam = np.maximum(cam, 0)                              # ReLU
    if cam.max() > 0:
        cam /= cam.max()
    return cam                                            # (h, w) in [0, 1]


def apply_heatmap_overlay(bgr_img: np.ndarray, heatmap: np.ndarray,
                          alpha: float = 0.45) -> np.ndarray:
    """
    Upscale heatmap → COLORMAP_JET → blend with the original image.

    Colour scale:
      Blue  = low activation (model ignored this area)
      Green = moderate
      Yellow = high
      Red   = hotspot → most likely fracture site

    Returns RGB uint8 at the same size as bgr_img.
    """
    h, w = bgr_img.shape[:2]
    hm_resized  = cv2.resize(heatmap, (w, h))
    hm_uint8    = np.uint8(255 * hm_resized)
    hm_colour   = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)   # BGR

    # Blend over original
    overlay_bgr = cv2.addWeighted(hm_colour, alpha, bgr_img, 1 - alpha, 0)
    return cv2.cvtColor(overlay_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)


def demo_heatmap(bgr_img: np.ndarray, is_fractured: bool) -> np.ndarray:
    """Synthesise a plausible heatmap when no model is loaded (demo mode)."""
    h, w = bgr_img.shape[:2]
    fake = np.zeros((h, w), dtype="float32")
    if is_fractured:
        cx, cy = int(w * 0.52), int(h * 0.38)
        rx, ry = int(w * 0.13), int(h * 0.09)
        Y, X   = np.ogrid[:h, :w]
        dist   = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        fake   = np.clip(1.0 - dist * 0.55, 0, 1).astype("float32")
    return apply_heatmap_overlay(bgr_img, fake)


# ── Full inference pipeline ───────────────────────────────────────────────────

def run_inference(image_bytes: bytes) -> dict:
    """
    1. Decode image (keep BGR for overlay blending)
    2. Preprocess exactly as original Streamlit app
    3. Classify with sigmoid threshold > 0.5
    4. Grad-CAM → heatmap overlay
    5. Return JSON-ready dict
    """
    bgr_img   = decode_bgr(image_bytes)
    img_array = preprocess_for_model(bgr_img)

    # ── Classification ────────────────────────────────────────────────────────
    if model is None:
        import random
        prediction   = round(random.uniform(0.1, 0.95), 4)
        is_fractured = prediction > 0.5
        is_demo      = True
    else:
        # Matches exactly: model.predict(processed_image)[0][0]
        prediction   = float(model.predict(img_array, verbose=0)[0][0])
        is_fractured = prediction > 0.5
        is_demo      = False

    label = "Fractured Bone" if is_fractured else "Normal Bone"

    # Confidence: distance from 0.5, mapped to a 0-1 scale
    # Raw prediction value is kept for display (matches original's f"{prediction:.2f}")
    confidence = prediction if is_fractured else (1.0 - prediction)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if grad_model is not None:
        heatmap     = compute_gradcam(img_array, is_fractured)
        overlay_img = apply_heatmap_overlay(bgr_img, heatmap)
    else:
        overlay_img = demo_heatmap(bgr_img, is_fractured)

    return {
        "label":        label,
        "confidence":   round(confidence, 4),
        "raw_score":    round(prediction, 4),   # original's "Confidence Score"
        "demo":         is_demo,
        "image_b64":    bytes_to_b64(image_bytes),
        "heatmap_b64":  to_b64_jpeg(overlay_img),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "xray" not in request.files:
        return jsonify({"error": "No file attached. Please upload an X-ray image."}), 400

    file = request.files["xray"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400
        return jsonify(run_inference(image_bytes)), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": "Unexpected error during analysis."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)