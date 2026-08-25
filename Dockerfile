FROM python:3.12-slim

WORKDIR /app

# Create the non-root user expected by Hugging Face Spaces
RUN useradd -m -u 1000 user

# Copy dependency file first for better Docker layer caching
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the complete project
COPY --chown=user:user . .

# Run as non-root user
USER user

# Hugging Face Spaces Docker port
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]