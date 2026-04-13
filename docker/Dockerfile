# Build from the project root (one level above docker/):
#   docker build -f docker/Dockerfile -t it-helpdesk-agents .
#
# Run the NiceGUI UI:
#   docker run --env-file .env -p 8000:8000 it-helpdesk-agents
#
# Run the evaluation benchmark:
#   docker run --env-file .env it-helpdesk-agents python evaluation/run_eval.py

FROM python:3.11-slim

# Keeps Python from generating .pyc files and forces stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build tools needed by some Python packages (e.g. scikit-learn wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (respects .dockerignore in the build context)
COPY . .

# NiceGUI listens on 8000 by default
EXPOSE 8000

# Default entrypoint: interactive UI
# Override CMD to run the benchmark instead:
#   docker run ... it-helpdesk-agents python evaluation/run_eval.py
CMD ["python", "ui/app.py"]
