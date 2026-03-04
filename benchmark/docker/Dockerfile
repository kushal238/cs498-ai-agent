FROM python:3.11-slim

LABEL description="Clinical workflow AI benchmark environment"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    langgraph \
    langchain \
    langchain-core \
    jsonschema \
    requests \
    rouge-score \
    scikit-learn \
    numpy \
    pytest

# Copy benchmark source (overridden at runtime by volume mount in compose)
COPY benchmark/ /app/benchmark/

# Default command: run the full evaluator
CMD ["python", "benchmark/runner/evaluate.py"]
