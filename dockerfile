FROM python:3.11-slim

# Install system dependencies needed for libraries like XGBoost or compiler requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to take advantage of Docker caching layers
COPY requirements.txt .

# Install dependencies smoothly
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure into the container workspace
COPY . .

# Ensure directory structures exist for data outputs and model storage
RUN mkdir -p data saved_model

# Make the pipeline execution script executable
RUN chmod +x run.sh

# Set the default entry command to execute the pipeline wrapper script
CMD ["./run.sh"]