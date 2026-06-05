web: gunicorn rmbg_server:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT
upscale: gunicorn upscale_server:app --workers 1 --timeout 180 --bind 0.0.0.0:${UPSCALE_PORT:-10001}
face: gunicorn face_server:app --workers 1 --timeout 180 --bind 0.0.0.0:${FACE_PORT:-10002}
