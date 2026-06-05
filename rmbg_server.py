from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io, base64, os

app = Flask(__name__)
CORS(app)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_DIMENSION   = 4096
MAX_BULK        = 20

print("Loading u2net model...")
session = new_session("u2net")
print("✅ Model loaded!")

def decode_image(data_uri):
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({len(raw)//1024} KB). Max is {MAX_IMAGE_BYTES//1024} KB.")
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    return img

def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def process_one(data_uri):
    return encode_image(remove(decode_image(data_uri), session=session))

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "u2net", "rembg": True})

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        return jsonify({"success": True, "image": process_one(data["image"])})
    except ValueError as e:
        return jsonify({"error": str(e)}), 413
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/remove-bg-bulk", methods=["POST"])
def remove_bg_bulk():
    try:
        data = request.get_json(silent=True)
        if not data or "images" not in data:
            return jsonify({"error": "Expected JSON body with key 'images' (list)"}), 400
        images = data["images"]
        if not isinstance(images, list) or len(images) == 0:
            return jsonify({"error": "'images' must be a non-empty list"}), 400
        if len(images) > MAX_BULK:
            return jsonify({"error": f"Max {MAX_BULK} images per request"}), 400
        results, errors = [], []
        for i, img_data in enumerate(images):
            try:
                results.append(process_one(img_data))
                errors.append(None)
            except Exception as e:
                results.append(None)
                errors.append(f"Image {i}: {str(e)}")
        return jsonify({"success": not any(errors), "images": results, "errors": errors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 RMBG Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
