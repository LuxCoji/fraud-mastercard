# Serving image for the fraud router.
#
# Models are NOT baked in. They are pulled from GCS at startup, so promoting a
# new model is a file upload rather than a rebuild - which is what makes the
# ModelRegistry's promote/rollback useful in production rather than only locally.
FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so a source change does not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn fastapi

COPY src/ ./src/
COPY deploy/fetch_models.py ./deploy/

ENV PYTHONUNBUFFERED=1 \
    MODEL_BUCKET="" \
    MODEL_PREFIX="fraud-mastercard/artifacts"

# Fetch models, then serve. Fetching at startup rather than at build time means
# the same image can run any model version.
CMD ["sh", "-c", "python deploy/fetch_models.py && uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
