from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io, base64, os, urllib.request, numpy as np

app = Flask(__name__)
CORS(app)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "models")
GFPGAN_URL      = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
GFPGAN_PATH     = os.path.join(MODEL_DIR, "GFPGANv1.3.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

_restorer = None

def get_restorer():
    global _restorer
    if _restorer is not None:
        return _restorer
    try:
        from gfpgan import GFPGANer
    except ImportError:
        raise RuntimeError("gfpgan not installed. Run: pip install gfpgan")
    if not os.path.exists(GFPGAN_PATH):
        print("Downloading GFPGAN model (~350MB)...")
        urllib.request.urlretrieve(GFPGAN_URL, GFPGAN_PATH)
        print("✅ GFPGAN model downloaded!")
    _restorer = GFPGANer(
        model_path=GFPGAN_PATH,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )
    print("✅ GFPGAN restorer ready!")
    return _restorer

def decode_image(data_uri):
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large.")
    return Image.open(io.BytesIO(raw)).convert("RGB")

def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "GFPGANv1.3"})

@app.route("/face-enhance", methods=["POST"])
def face_enhance():
    try:
        restorer = get_restorer()

        if request.files.get("image"):
            raw = request.files["image"].read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            data = request.get_json(silent=True)
            if not data or "image" not in data:
                return jsonify({"error": "No image provided"}), 400
            img = decode_image(data["image"])

        img_np = np.array(img)[:, :, ::-1]  # RGB → BGR for GFPGAN

        _, _, output = restorer.enhance(
            img_np, has_aligned=False, only_center_face=False, paste_back=True
        )

        result = Image.fromarray(output[:, :, ::-1])  # BGR → RGB

        if request.files.get("image"):
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype="image/png", download_name="face_enhanced.png")

        return jsonify({"success": True, "image": encode_image(result)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 413
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("FACE_PORT", 10002))
    print(f"🚀 Face Enhance Server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
