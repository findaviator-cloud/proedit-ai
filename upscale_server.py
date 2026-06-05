from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io, base64, os, urllib.request

app = Flask(__name__)
CORS(app)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_DIMENSION   = 2048
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "models")
ESRGAN_URL      = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
ESRGAN_PATH     = os.path.join(MODEL_DIR, "realesr-general-x4v3.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# Lazy-load upscaler on first request
_upsampler = None

def get_upsampler():
    global _upsampler
    if _upsampler is not None:
        return _upsampler
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise RuntimeError("realesrgan not installed. Run: pip install realesrgan basicsr")
    if not os.path.exists(ESRGAN_PATH):
        print("Downloading RealESRGAN model (~65MB)...")
        urllib.request.urlretrieve(ESRGAN_URL, ESRGAN_PATH)
        print("✅ ESRGAN model downloaded!")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4)
    _upsampler = RealESRGANer(
        scale=4, model_path=ESRGAN_PATH, model=model,
        tile=512, tile_pad=10, pre_pad=0, half=False
    )
    print("✅ ESRGAN upsampler ready!")
    return _upsampler

def decode_image(data_uri):
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({len(raw)//1024} KB). Max {MAX_IMAGE_BYTES//1024} KB.")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    return img

def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "realesrgan-x4v3"})

@app.route("/upscale", methods=["POST"])
def upscale():
    try:
        import numpy as np
        upsampler = get_upsampler()

        # Multipart
        if request.files.get("image"):
            raw = request.files["image"].read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            data = request.get_json(silent=True)
            if not data or "image" not in data:
                return jsonify({"error": "No image provided"}), 400
            img = decode_image(data["image"])

        scale = min(int(request.args.get("scale", 4)), 4)

        img_np = np.array(img)
        output, _ = upsampler.enhance(img_np, outscale=scale)
        result = Image.fromarray(output)

        if request.files.get("image"):
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype="image/png", download_name="upscaled.png")

        return jsonify({"success": True, "image": encode_image(result),
                        "original_size": list(img.size),
                        "output_size": list(result.size)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 413
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("UPSCALE_PORT", 10001))
    print(f"🚀 Upscale Server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
