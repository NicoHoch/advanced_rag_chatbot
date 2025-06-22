# Dockerfile

FROM python:3.12-slim
ENV ENV=prod

# Install poppler-utils and dependencies
RUN apt-get update && \
    apt-get install -y poppler-utils gcc libpoppler-cpp-dev && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 80

# Run the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]