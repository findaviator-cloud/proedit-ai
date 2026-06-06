FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir basicsr==1.4.2 --no-build-isolation
RUN pip install --no-cache-dir realesrgan==0.3.0 gfpgan==1.3.8 facexlib==0.3.0
RUN pip install --no-cache-dir flask==3.0.3 flask-cors==4.0.1 gunicorn==21.2.0 pillow==10.4.0 opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir "rembg[cpu]==2.0.57"
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py

COPY . .

EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "rmbg_server:app", "--workers", "1", "--timeout", "180", "--bind", "0.0.0.0:7860"]
