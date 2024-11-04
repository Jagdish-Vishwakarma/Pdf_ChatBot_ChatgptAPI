# Use the official Python image with a specific version compatible with Streamlit
FROM python:3.9-slim

# Install system dependencies for pillow
RUN apt-get update && \
    apt-get install -y libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the app files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py"]
