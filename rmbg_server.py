from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io, base64, os, zipfile, concurrent.futures

app = Flask(__name__)
CORS(app)

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB
MAX_DIMENSION   = 4096
MAX_BULK        = 20
MAX_WORKERS     = 4

print("Loading birefnet-general model...")
session = new_session("birefnet-general")
print("✅ Model loaded!")

# ── helpers ──────────────────────────────────────────────────────────────────

def decode_image(data_uri: str) -> Image.Image:
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({len(raw)//1024} KB). Max {MAX_IMAGE_BYTES//1024} KB.")
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    return img

def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def process_file_bytes(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    result = remove(img, session=session)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()

def process_one(data_uri: str) -> str:
    img = decode_image(data_uri)
    result = remove(img, session=session)
    return encode_image(result)

# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "birefnet-general", "rembg": True})

# Single image — base64 JSON (legacy + frontend compat)
@app.route("/remove-bg", methods=["POST"])
@app.route("/removebg",  methods=["POST"])   # frontend alias fix
def remove_bg():
    try:
        # Support multipart upload
        if request.files.get("image"):
            raw = request.files["image"].read()
            result_bytes = process_file_bytes(raw)
            return send_file(
                io.BytesIO(result_bytes),
                mimetype="image/png",
                as_attachment=False,
                download_name="result.png"
            )
        # Fallback: base64 JSON
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        return jsonify({"success": True, "image": process_one(data["image"])})
    except ValueError as e:
        return jsonify({"error": str(e)}), 413
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Bulk — base64 JSON, parallel processing
@app.route("/remove-bg-bulk", methods=["POST"])
@app.route("/removebg-bulk", methods=["POST"])  # alias
def remove_bg_bulk():
    try:
        data = request.get_json(silent=True)
        if not data or "images" not in data:
            return jsonify({"error": "Expected {'images': [...]}"}), 400
        images = data["images"]
        if not isinstance(images, list) or len(images) == 0:
            return jsonify({"error": "'images' must be a non-empty list"}), 400
        if len(images) > MAX_BULK:
            return jsonify({"error": f"Max {MAX_BULK} images per request"}), 400

        results = [None] * len(images)
        errors  = [None] * len(images)

        def _process(idx, uri):
            try:
                results[idx] = process_one(uri)
            except Exception as e:
                errors[idx] = f"Image {idx}: {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(_process, i, uri) for i, uri in enumerate(images)]
            concurrent.futures.wait(futures)

        return jsonify({
            "success": not any(errors),
            "images":  results,
            "errors":  errors,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Bulk multipart → ZIP download
@app.route("/remove-bg-bulk-zip", methods=["POST"])
def remove_bg_bulk_zip():
    try:
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        if len(files) > MAX_BULK:
            return jsonify({"error": f"Max {MAX_BULK} images per request"}), 400

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            def _process_file(idx, f):
                raw = f.read()
                name = os.path.splitext(f.filename or f"image_{idx}")[0] + "_nobg.png"
                try:
                    result = process_file_bytes(raw)
                    return (name, result, None)
                except Exception as e:
                    return (name, None, str(e))

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(_process_file, i, f) for i, f in enumerate(files)]
                for fut in concurrent.futures.as_completed(futures):
                    name, result, err = fut.result()
                    if result:
                        zf.writestr(name, result)

        zip_buf.seek(0)
        return send_file(
            zip_buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="removed_backgrounds.zip"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 RMBG Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
