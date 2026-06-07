from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io, base64, os, zipfile, concurrent.futures, threading, urllib.request, numpy as np

app = Flask(__name__)
CORS(app)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_DIMENSION   = 4096
MAX_BULK        = 20
MAX_WORKERS     = 2
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading birefnet-general model...")
session = new_session("birefnet-general")
print("✅ Model loaded!")

# ── ESRGAN lazy load ──────────────────────────────────────────────────────────
_upsampler = None
ESRGAN_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
ESRGAN_PATH = os.path.join(MODEL_DIR, "realesr-general-x4v3.pth")

def get_upsampler():
    global _upsampler
    if _upsampler: return _upsampler
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    if not os.path.exists(ESRGAN_PATH):
        print("Downloading ESRGAN model...")
        urllib.request.urlretrieve(ESRGAN_URL, ESRGAN_PATH)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    _upsampler = RealESRGANer(scale=4, model_path=ESRGAN_PATH, model=model,
                               tile=256, tile_pad=10, pre_pad=0, half=False)
    print("✅ ESRGAN ready!")
    return _upsampler

# ── GFPGAN lazy load ──────────────────────────────────────────────────────────
_restorer = None
GFPGAN_URL  = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
GFPGAN_PATH = os.path.join(MODEL_DIR, "GFPGANv1.3.pth")

def get_restorer():
    global _restorer
    if _restorer: return _restorer
    from gfpgan import GFPGANer
    if not os.path.exists(GFPGAN_PATH):
        print("Downloading GFPGAN model...")
        urllib.request.urlretrieve(GFPGAN_URL, GFPGAN_PATH)
    _restorer = GFPGANer(model_path=GFPGAN_PATH, upscale=2,
                          arch="clean", channel_multiplier=2, bg_upsampler=None)
    print("✅ GFPGAN ready!")
    return _restorer

# ── helpers ───────────────────────────────────────────────────────────────────
def decode_image(data_uri):
    if "," in data_uri: data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({len(raw)//1024}KB). Max {MAX_IMAGE_BYTES//1024}KB.")
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    return img

def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def process_one(data_uri):
    img = decode_image(data_uri)
    return encode_image(remove(img, session=session))

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","model":"birefnet-general","rembg":True})

@app.route("/remove-bg", methods=["POST"])
@app.route("/removebg",  methods=["POST"])
def remove_bg():
    try:
        if request.files.get("image"):
            raw = request.files["image"].read()
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
            result = remove(img, session=session)
            buf = io.BytesIO(); result.save(buf, format="PNG"); buf.seek(0)
            return send_file(buf, mimetype="image/png", download_name="result.png")
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error":"No image provided"}), 400
        return jsonify({"success":True,"image":process_one(data["image"])})
    except ValueError as e: return jsonify({"error":str(e)}), 413
    except Exception as e:  return jsonify({"error":str(e)}), 500

@app.route("/remove-bg-bulk", methods=["POST"])
@app.route("/removebg-bulk", methods=["POST"])
def remove_bg_bulk():
    try:
        data = request.get_json(silent=True)
        if not data or "images" not in data:
            return jsonify({"error":"Expected {'images':[...]}"}), 400
        images = data["images"]
        if not isinstance(images, list) or len(images)==0:
            return jsonify({"error":"'images' must be non-empty list"}), 400
        if len(images) > MAX_BULK:
            return jsonify({"error":f"Max {MAX_BULK} images"}), 400
        results=[None]*len(images); errors=[None]*len(images)
        def _process(idx, uri):
            try: results[idx]=process_one(uri)
            except Exception as e: errors[idx]=f"Image {idx}: {str(e)}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            concurrent.futures.wait([ex.submit(_process,i,uri) for i,uri in enumerate(images)])
        return jsonify({"success":not any(errors),"images":results,"errors":errors})
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/remove-bg-bulk-zip", methods=["POST"])
def remove_bg_bulk_zip():
    try:
        files = request.files.getlist("images")
        if not files: return jsonify({"error":"No files"}), 400
        if len(files) > MAX_BULK: return jsonify({"error":f"Max {MAX_BULK}"}), 400
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, f in enumerate(files):
                raw = f.read()
                name = os.path.splitext(f.filename or f"image_{i}")[0]+"_nobg.png"
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGBA")
                    result = remove(img, session=session)
                    buf = io.BytesIO(); result.save(buf, format="PNG")
                    zf.writestr(name, buf.getvalue())
                except: pass
        zip_buf.seek(0)
        return send_file(zip_buf, mimetype="application/zip",
                        as_attachment=True, download_name="removed_backgrounds.zip")
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/upscale", methods=["POST"])
def upscale():
    try:
        upsampler = get_upsampler()
        if request.files.get("image"):
            raw = request.files["image"].read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            data = request.get_json(silent=True)
            if not data or "image" not in data:
                return jsonify({"error":"No image"}), 400
            img = decode_image(data["image"]).convert("RGB")
        scale = min(int(request.args.get("scale", data.get("scale",4) if not request.files.get("image") else 4)), 4)
        img_np = np.array(img)
        output, _ = upsampler.enhance(img_np, outscale=scale)
        result = Image.fromarray(output)
        return jsonify({"success":True,"image":encode_image(result),
                       "original_size":list(img.size),"output_size":list(result.size)})
    except Exception as e: return jsonify({"error":str(e)}), 500

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
                return jsonify({"error":"No image"}), 400
            img = decode_image(data["image"]).convert("RGB")
        img_np = np.array(img)[:,:,::-1]
        _, _, output = restorer.enhance(img_np, has_aligned=False,
                                         only_center_face=False, paste_back=True)
        result = Image.fromarray(output[:,:,::-1])
        return jsonify({"success":True,"image":encode_image(result)})
    except Exception as e: return jsonify({"error":str(e)}), 500

# ── warmup ────────────────────────────────────────────────────────────────────
def _warmup():
    try:
        dummy = Image.new("RGBA",(64,64),(255,0,0,255))
        remove(dummy, session=session)
        print("✅ Warmup done!")
    except Exception as e:
        print(f"⚠️ Warmup failed: {e}")

threading.Thread(target=_warmup, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    print(f"🚀 Server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
