# Accept a base image
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Accept a component name (optional, for labeling)
ARG COMPONENT_NAME

# Switch to root (UBI images default to non-root)
USER 0

# Install system dependencies (Milvus SDK sometimes needs gcc during build)
RUN dnf install -y git gcc && dnf clean all

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=DEBUG

# Set working directory
WORKDIR /app

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh && \
    install -Dm755 "$HOME/.local/bin/uv" /usr/local/bin/uv

# Copy only the files needed to install dependencies
COPY pyproject.toml .
COPY README.md .

# Install only runtime dependencies into the system Python
RUN echo "Using Python 🐍 from: $(which python)" && \
    python --version && \
    uv pip install --python $(which python) --system .

# Copy the full application code
COPY . .

# Now switch to a non-root user for runtime
USER 1001

# Expose the FastAPI port
EXPOSE 7777

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777", "--log-level", "debug", "--reload"]
