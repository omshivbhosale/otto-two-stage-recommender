FROM python:3.12-slim

# LightGBM needs the OpenMP runtime on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: the server has no GPU, and the CPU build is far smaller
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY artifacts/ artifacts/
COPY models/session_gru_epoch1.pt models/ranker_y_click.txt models/ranker_y_cart.txt models/ranker_y_order.txt models/

# one thread each for torch and lightgbm (see the note in src/serve.py)
ENV OMP_NUM_THREADS=1

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]