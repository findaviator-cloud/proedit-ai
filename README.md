# ProEdit AI — Professional Image Studio

> © 2026 Trishika AI — All Rights Reserved

A powerful single-file AI image editor with background removal, bulk processing, smart crop, filters, and more.

## Features
- 🤖 AI Background Removal (u2net / isnet-general-use)
- 🖼️ Bulk Processing
- ✂️ Smart Crop (auto white/transparent trim)
- 🎨 Filters (grayscale, sepia, invert, blur, brightness, contrast)
- 🔄 Rotate, Flip, Resize
- 💾 Export PNG / JPG / WebP
- 📐 Crop tool with drag selection

## Setup

### Requirements
- Python 3.10+
- WSL / Linux / Mac

### Install
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
# Terminal 1 — AI Server
python3 rmbg_server.py

# Terminal 2 — HTML Server  
python3 -m http.server 8080
```

Open browser: `http://localhost:8080/ai-image-editor-pro.html`

## License
Copyright (c) 2026 Trishika AI. All Rights Reserved.
Unauthorized use is strictly prohibited.
