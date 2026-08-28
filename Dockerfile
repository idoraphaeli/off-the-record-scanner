# Build recipe for hosts that build from the repository ROOT.
#
# server/Dockerfile is the same image built from inside server/, which is what
# Render does (its blueprint sets dockerContext: ./server). Google Cloud Build,
# driven from the Cloud Run console, does not offer that choice — it hands the
# whole repository to the Dockerfile — so the paths here are written from the
# root instead. The two must be kept in step; they install the same things.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Two, because Cloud Run is deployed with two CPUs and OpenCV can use them.
    # scanner/__init__.py reads this, so the host decides, not the code.
    OPENCV_NUM_THREADS=2 \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/scanner ./scanner
COPY server/app.py server/capture.html ./

# Cloud Run injects PORT and expects the container to listen on it; 8080 is its
# default and a sensible one to fall back on elsewhere.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
