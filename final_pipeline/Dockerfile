FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /workspace/final_pipeline
COPY requirements.txt requirements-gpu.txt ./
RUN python -m pip install --no-cache-dir -r requirements-gpu.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN python -m pip install --no-deps -e .
ENV PYTHONUNBUFFERED=1
CMD ["bash"]
