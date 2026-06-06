FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "rmbg_server:app", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:7860"]
