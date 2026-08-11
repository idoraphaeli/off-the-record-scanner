FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # OpenCV spawns a thread pool per operation; on a 1-2 core instance that
    # costs more in contention than it saves, and it inflates memory.
    OPENCV_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scanner ./scanner
COPY app.py capture.html ./

# Render (and most hosts) inject the port to listen on; default for local runs.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
