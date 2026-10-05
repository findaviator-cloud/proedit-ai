import io
import os
import time
import json
import base64
import logging
import zipfile
import threading
import urllib.request
import concurrent.futures
from collections import defaultdict, deque

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rembg import remove, new_session

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("proedit-ai")

# ── config (env-overridable) ────────────────────────────────────────────────
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_MB", 20)) * 1024 * 1024
MAX_DIMENSION   = int(os.environ.get("MAX_DIMENSION", 4096))
MAX_BULK        = int(os.environ.get("MAX_BULK", 20))
MAX_WORKERS     = int(os.environ.get("MAX_WORKERS", 2))
RATE_LIMIT      = int(os.environ.get("RATE_LIMIT_PER_MIN", 30))  # requests/min/IP, 0 disables
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# CORS: comma-separated list of allowed origins in prod. "*" only if explicitly set.
_allowed = os.environ.get("ALLOWED_ORIGINS", "").strip()
if _allowed:
    ALLOWED_ORIGINS = [o.strip() for o in _allowed.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = "*"
    log.warning("ALLOWED_ORIGINS not set - CORS is wide open (*). Set it in production.")

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS)

# ── simple in-memory per-IP rate limiter (no extra dependency) ────────────────
# Fine for a single-process/single-instance deploy. Swap for a Redis-backed
# limiter (e.g. flask-limiter) if you scale to multiple workers/instances.
_hits = defaultdict(deque)
_hits_lock = threading.Lock()

def rate_limited(ip):
    if RATE_LIMIT <= 0:
        return False
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return True
        q.append(now)
        return False

@app.before_request
def _check_rate_limit():
    if request.path == "/health":
        return None
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if rate_limited(ip):
        return jsonify({"error": "Rate limit exceeded. Try again in a minute."}), 429

def guard_content_length(max_bytes):
    """Reject oversized requests before we read the body."""
    cl = request.content_length
    if cl is not None and cl > max_bytes:
        raise ValueError(f"Request too large ({cl // 1024 // 1024}MB). Max {max_bytes // 1024 // 1024}MB.")

log.info("Loading birefnet-general model...")
try:
    session = new_session("birefnet-general")
    log.info("Model loaded.")
except Exception:
    log.critical("Failed to load rembg model at startup", exc_info=True)
    raise

# ── ESRGAN lazy load ──────────────────────────────────────────────────────────
_upsampler = None
ESRGAN_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
ESRGAN_PATH = os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth")

def get_upsampler():
    global _upsampler
    if _upsampler:
        return _upsampler
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    if not os.path.exists(ESRGAN_PATH):
        log.info("Downloading RealESRGAN x4plus model...")
        urllib.request.urlretrieve(ESRGAN_URL, ESRGAN_PATH)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    _upsampler = RealESRGANer(scale=4, model_path=ESRGAN_PATH, model=model,
                               tile=256, tile_pad=10, pre_pad=0, half=False)
    log.info("ESRGAN ready.")
    return _upsampler

# ── GFPGAN lazy load ──────────────────────────────────────────────────────────
_restorer = None
GFPGAN_URL  = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
GFPGAN_PATH = os.path.join(MODEL_DIR, "GFPGANv1.3.pth")

def get_restorer():
    global _restorer
    if _restorer:
        return _restorer
    from gfpgan import GFPGANer
    if not os.path.exists(GFPGAN_PATH):
        log.info("Downloading GFPGAN model...")
        urllib.request.urlretrieve(GFPGAN_URL, GFPGAN_PATH)
    _restorer = GFPGANer(model_path=GFPGAN_PATH, upscale=2,
                          arch="clean", channel_multiplier=2, bg_upsampler=None)
    log.info("GFPGAN ready.")
    return _restorer

# ── helpers ───────────────────────────────────────────────────────────────────
def decode_image(data_uri):
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
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

def load_input_image(mode="RGB"):
    """Shared loader for multipart-file or JSON-base64 input.
    Returns (image, request_json_or_None)."""
    if request.files.get("image"):
        raw = request.files["image"].read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Image too large")
        return Image.open(io.BytesIO(raw)).convert(mode), None
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        raise ValueError("No image provided")
    return decode_image(data["image"]).convert(mode), data

# ── Smart document auto-crop (edge detection + perspective correction) ──────
DOC_MIN_AREA_FRAC = float(os.environ.get("DOC_MIN_AREA_FRAC", 0.08))

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def _four_point_transform(cv2mod, image_bgr, pts):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 1)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 1)
    dst = np.array([[0, 0], [max_width - 1, 0],
                     [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
    m = cv2mod.getPerspectiveTransform(rect, dst)
    return cv2mod.warpPerspective(image_bgr, m, (max_width, max_height))

def _find_document_contour_brightness(cv2mod, image_bgr, invert, min_area_frac,
                                       max_area_frac=0.98, min_rectangularity=0.55):
    """Brightness-mask based detection (Otsu threshold + morphology + convex
    hull + min-area-rect) — much more robust than pure edge detection against
    real-world photos: shadows, paper folds/creases, printed text near the
    edges, and low-contrast backgrounds all break clean 4-point edge contours,
    but the document is still reliably the largest bright (or, if invert=True,
    darkest — e.g. a dark phone screen on a light cloth) region in the frame.

    Returns (box, area_frac, rectangularity) or None.

    Safety checks:
    - rectangularity (contour area / its min-area-rect area) must be
      reasonably high — a true document/phone is close to its own bounding
      rectangle; a spurious sliver of background (wood grain, a shadow) is
      usually far from rectangular and gets filtered out here.
    - if the region touches all 4 frame edges, we've likely captured
      "everything" (e.g. a page atop a stack of similarly-toned pages, or a
      card photographed on a page of nearly the same brightness so the mask
      merges them into one edge-to-edge blob) rather than one distinct
      object — bail out rather than return a barely-cropped, falsely-
      confident result. A handful of genuine "object fills the whole frame"
      photos (e.g. an extreme close-up of a phone) will be rejected by this
      too; that's an accepted trade-off, since the alternative — trying to
      allow those back in — reliably reopens the far more common
      card-on-a-page false-positive instead."""
    h, w = image_bgr.shape[:2]
    target_h = 600
    ratio = h / float(target_h) if h > target_h else 1.0
    resized = cv2mod.resize(image_bgr, (max(1, int(w / ratio)), target_h)) if ratio != 1.0 else image_bgr.copy()
    rh, rw = resized.shape[:2]

    gray = cv2mod.cvtColor(resized, cv2mod.COLOR_BGR2GRAY)
    blur = cv2mod.GaussianBlur(gray, (7, 7), 0)
    thresh_type = cv2mod.THRESH_BINARY_INV if invert else cv2mod.THRESH_BINARY
    _, mask = cv2mod.threshold(blur, 0, 255, thresh_type + cv2mod.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2mod.morphologyEx(mask, cv2mod.MORPH_CLOSE, kernel)
    mask = cv2mod.morphologyEx(mask, cv2mod.MORPH_OPEN, kernel)

    contours, _ = cv2mod.findContours(mask, cv2mod.RETR_EXTERNAL, cv2mod.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2mod.contourArea)
    img_area = rh * rw
    area_frac = cv2mod.contourArea(biggest) / img_area
    if area_frac < min_area_frac or area_frac > max_area_frac:
        return None

    rect = cv2mod.minAreaRect(biggest)
    rect_area = rect[1][0] * rect[1][1]
    rectangularity = cv2mod.contourArea(biggest) / max(rect_area, 1)
    if rectangularity < min_rectangularity:
        return None

    x, y, bw, bh = cv2mod.boundingRect(biggest)
    margin = 0.03
    mx, my = rw * margin, rh * margin
    edges_touched = sum([
        x <= mx, y <= my, (x + bw) >= (rw - mx), (y + bh) >= (rh - my)
    ])
    if edges_touched == 4:
        return None

    hull = cv2mod.convexHull(biggest)
    rect2 = cv2mod.minAreaRect(hull)
    box = cv2mod.boxPoints(rect2)
    return (box * ratio).astype("float32"), area_frac, rectangularity

def _find_document_contour_edges(cv2mod, image_bgr, min_area_frac=0.015, max_area_frac=0.5,
                                  min_rectangularity=0.5, min_aspect=0.15):
    """Sensitive-Canny edge-based fallback for when the brightness method
    fails — specifically helps the case of a small, distinct document (a
    receipt, ID card) sitting on a much larger page/surface of nearly the
    SAME brightness, where there's no brightness contrast to exploit but the
    small object's printed border / drop-shadow is still a real edge.

    Tries candidate contours largest-first and returns the first one that
    passes rectangularity and aspect-ratio sanity checks — Canny edges are
    noisier than a brightness mask, so without this a stray strip of
    background texture can outrank the real (slightly smaller) document."""
    h, w = image_bgr.shape[:2]
    target_h = 800
    ratio = h / float(target_h) if h > target_h else 1.0
    resized = cv2mod.resize(image_bgr, (max(1, int(w / ratio)), target_h)) if ratio != 1.0 else image_bgr.copy()
    rh, rw = resized.shape[:2]

    gray = cv2mod.cvtColor(resized, cv2mod.COLOR_BGR2GRAY)
    blur = cv2mod.GaussianBlur(gray, (3, 3), 0)
    edged = cv2mod.Canny(blur, 15, 60)
    edged = cv2mod.dilate(edged, np.ones((5, 5), np.uint8), iterations=2)
    edged = cv2mod.morphologyEx(edged, cv2mod.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2mod.findContours(edged, cv2mod.RETR_EXTERNAL, cv2mod.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = rh * rw
    candidates = sorted(
        [c for c in contours if min_area_frac * img_area <= cv2mod.contourArea(c) <= max_area_frac * img_area],
        key=cv2mod.contourArea, reverse=True)

    margin = 0.02
    mx, my = rw * margin, rh * margin
    for c in candidates:
        rect = cv2mod.minAreaRect(c)
        (rw_, rh_) = rect[1]
        if min(rw_, rh_) < 1:
            continue
        rectangularity = cv2mod.contourArea(c) / max(rw_ * rh_, 1)
        aspect = min(rw_, rh_) / max(rw_, rh_)
        if rectangularity < min_rectangularity or aspect < min_aspect:
            continue
        x, y, bw, bh = cv2mod.boundingRect(c)
        edges_touched = sum([
            x <= mx, y <= my, (x + bw) >= (rw - mx), (y + bh) >= (rh - my)
        ])
        if edges_touched == 4:
            continue
        hull = cv2mod.convexHull(c)
        rect2 = cv2mod.minAreaRect(hull)
        box = cv2mod.boxPoints(rect2)
        return (box * ratio).astype("float32")
    return None

def _find_document_contour(cv2mod, image_bgr):
    """Try brightness detection in both polarities — document brighter than
    its surroundings (the common case: paper on a table) AND document darker
    than its surroundings (e.g. a dark phone screen on light cloth) — and
    keep whichever candidate scores highest on (area * rectangularity), i.e.
    the largest, cleanest rectangle. The inverted polarity requires a much
    larger minimum area (0.30 vs 0.08) since small dark regions (shadows,
    wood grain) are far more likely to be false positives than small bright
    ones. If neither brightness polarity finds anything confident, fall back
    to edge detection (small document on a similarly-bright larger surface)."""
    candidates = []
    c = _find_document_contour_brightness(cv2mod, image_bgr, invert=False, min_area_frac=DOC_MIN_AREA_FRAC)
    if c is not None:
        candidates.append(c)
    c = _find_document_contour_brightness(cv2mod, image_bgr, invert=True, min_area_frac=0.30)
    if c is not None:
        candidates.append(c)
    if candidates:
        box, _, _ = max(candidates, key=lambda c: c[1] * c[2])
        return box
    return _find_document_contour_edges(cv2mod, image_bgr)

@app.route("/auto-crop-document", methods=["POST"])
def auto_crop_document():
    try:
        guard_content_length(MAX_IMAGE_BYTES)
        import cv2
        img, _ = load_input_image("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        corners = _find_document_contour(cv2, img_bgr)
        if corners is None:
            return jsonify({"success": False, "detected": False,
                             "error": "Document boundary not detected — try manual crop"})

        warped_bgr = _four_point_transform(cv2, img_bgr, corners)
        result = Image.fromarray(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB))
        return jsonify({"success": True, "detected": True, "image": encode_image(result),
                         "output_size": list(result.size)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400 if "No image" in str(e) else 413
    except Exception:
        log.exception("auto_crop_document failed")
        return jsonify({"error": "Internal error auto-cropping document"}), 500

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "birefnet-general", "rembg": True})

@app.route("/remove-bg", methods=["POST"])
@app.route("/removebg",  methods=["POST"])
def remove_bg():
    try:
        guard_content_length(MAX_IMAGE_BYTES)
        if request.files.get("image"):
            raw = request.files["image"].read()
            if len(raw) > MAX_IMAGE_BYTES:
                return jsonify({"error": "Image too large"}), 413
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
            result = remove(img, session=session)
            buf = io.BytesIO(); result.save(buf, format="PNG"); buf.seek(0)
            return send_file(buf, mimetype="image/png", download_name="result.png")
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        return jsonify({"success": True, "image": process_one(data["image"])})
    except ValueError as e:
        return jsonify({"error": str(e)}), 413
    except Exception:
        log.exception("remove_bg failed")
        return jsonify({"error": "Internal error processing image"}), 500

@app.route("/remove-bg-bulk", methods=["POST"])
@app.route("/removebg-bulk", methods=["POST"])
def remove_bg_bulk():
    try:
        guard_content_length(MAX_IMAGE_BYTES * MAX_BULK)
        data = request.get_json(silent=True)
        if not data or "images" not in data:
            return jsonify({"error": "Expected {'images':[...]}"}), 400
        images = data["images"]
        if not isinstance(images, list) or len(images) == 0:
            return jsonify({"error": "'images' must be non-empty list"}), 400
        if len(images) > MAX_BULK:
            return jsonify({"error": f"Max {MAX_BULK} images"}), 400
        results = [None] * len(images)
        errors = [None] * len(images)

        def _process(idx, uri):
            try:
                results[idx] = process_one(uri)
            except Exception as e:
                errors[idx] = f"Image {idx}: {e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            concurrent.futures.wait([ex.submit(_process, i, uri) for i, uri in enumerate(images)])
        return jsonify({"success": not any(errors), "images": results, "errors": errors})
    except ValueError as e:
        return jsonify({"error": str(e)}), 413
    except Exception:
        log.exception("remove_bg_bulk failed")
        return jsonify({"error": "Internal error processing images"}), 500

@app.route("/remove-bg-bulk-zip", methods=["POST"])
def remove_bg_bulk_zip():
    try:
        guard_content_length(MAX_IMAGE_BYTES * MAX_BULK)
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No files"}), 400
        if len(files) > MAX_BULK:
            return jsonify({"error": f"Max {MAX_BULK}"}), 400
        zip_buf = io.BytesIO()
        failures = []
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, f in enumerate(files):
                name = os.path.splitext(f.filename or f"image_{i}")[0] + "_nobg.png"
                try:
                    raw = f.read()
                    if len(raw) > MAX_IMAGE_BYTES:
                        raise ValueError("file too large")
                    img = Image.open(io.BytesIO(raw)).convert("RGBA")
                    result = remove(img, session=session)
                    buf = io.BytesIO(); result.save(buf, format="PNG")
                    zf.writestr(name, buf.getvalue())
                except Exception as e:
                    failures.append({"file": f.filename or f"image_{i}", "error": str(e)})
                    log.warning("bulk-zip: failed on %s: %s", f.filename, e)
            if failures:
                zf.writestr("_errors.json", json.dumps(failures, indent=2))
        zip_buf.seek(0)
        return send_file(zip_buf, mimetype="application/zip",
                          as_attachment=True, download_name="removed_backgrounds.zip")
    except ValueError as e:
        return jsonify({"error": str(e)}), 413
    except Exception:
        log.exception("remove_bg_bulk_zip failed")
        return jsonify({"error": "Internal error processing images"}), 500

@app.route("/upscale", methods=["POST"])
def upscale():
    try:
        guard_content_length(MAX_IMAGE_BYTES)
        upsampler = get_upsampler()
        img, body = load_input_image("RGB")
        raw_scale = request.args.get("scale") or (body.get("scale") if body else None) or 4
        scale = min(int(raw_scale), 4)
        img_np = np.array(img)
        output, _ = upsampler.enhance(img_np, outscale=scale)
        result = Image.fromarray(output)
        return jsonify({"success": True, "image": encode_image(result),
                         "original_size": list(img.size), "output_size": list(result.size)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400 if "No image" in str(e) else 413
    except Exception:
        log.exception("upscale failed")
        return jsonify({"error": "Internal error upscaling image"}), 500

@app.route("/face-enhance", methods=["POST"])
def face_enhance():
    try:
        guard_content_length(MAX_IMAGE_BYTES)
        restorer = get_restorer()
        img, _ = load_input_image("RGB")
        img_np = np.array(img)[:, :, ::-1]
        _, _, output = restorer.enhance(img_np, has_aligned=False,
                                         only_center_face=False, paste_back=True)
        result = Image.fromarray(output[:, :, ::-1])
        return jsonify({"success": True, "image": encode_image(result)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400 if "No image" in str(e) else 413
    except Exception:
        log.exception("face_enhance failed")
        return jsonify({"error": "Internal error enhancing face"}), 500

# ── Magic Eraser (OpenCV Inpainting) ─────────────────────────────────────────
@app.route("/inpaint", methods=["POST"])
def inpaint():
    try:
        guard_content_length(MAX_IMAGE_BYTES * 2)  # image + mask
        import cv2
        data = request.get_json(silent=True)
        if not data or "image" not in data or "mask" not in data:
            return jsonify({"error": "image and mask required"}), 400

        img = decode_image(data["image"]).convert("RGB")
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        mask_uri = data["mask"]
        if "," in mask_uri:
            mask_uri = mask_uri.split(",", 1)[1]
        mask_raw = base64.b64decode(mask_uri)
        if len(mask_raw) > MAX_IMAGE_BYTES:
            raise ValueError("Mask too large")
        mask_img = Image.open(io.BytesIO(mask_raw)).convert("L")
        mask_img = mask_img.resize((img.width, img.height), Image.LANCZOS)
        mask_np = np.array(mask_img)

        _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        radius = int(data.get("radius", 3))
        radius = max(1, min(radius, 15))  # sanity-bound user input
        result_bgr = cv2.inpaint(img_bgr, mask_bin, radius, cv2.INPAINT_TELEA)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(result_rgb)

        return jsonify({"success": True, "image": encode_image(result)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 413
    except Exception:
        log.exception("inpaint failed")
        return jsonify({"error": "Internal error inpainting image"}), 500

# ── warmup ────────────────────────────────────────────────────────────────────
def _warmup():
    try:
        dummy = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        remove(dummy, session=session)
        log.info("Warmup done.")
    except Exception:
        log.warning("Warmup failed", exc_info=True)

threading.Thread(target=_warmup, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log.info("Server starting on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
