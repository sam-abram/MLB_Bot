# Stage 1: Build React frontend
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python backend
FROM python:3.11-slim
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY app/ ./app/
COPY train_model2.py .

# Seed/fallback model artifacts baked into the image. The live model is synced
# from S3 at runtime (app/model_loader.sync_from_s3); these COPYs only provide a
# starting model so the container can serve before the first S3 sync completes.
COPY model_artifacts/ ./model_artifacts/
COPY preprocessed/ ./preprocessed/

# Copy built frontend from stage 1
COPY --from=frontend-build /frontend/build ./static/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
