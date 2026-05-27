from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Load u2net session once at startup
print("Loading u2net model...")
session = new_session("isnet-general-use")
print("✅ Model loaded!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "u2net", "rembg": True})

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # Remove background
        output_image = remove(input_image, session=session)

        # Encode output to base64
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        output_b64 = base64.b64encode(output_buffer.read()).decode('utf-8')

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{output_b64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/remove-bg-file', methods=['POST'])
def remove_bg_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        input_image = Image.open(file.stream).convert("RGBA")

        # Remove background
        output_image = remove(input_image, session=session)

        # Return as PNG file
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return send_file(output_buffer, mimetype='image/png', 
                        download_name='removed_bg.png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 RMBG Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
