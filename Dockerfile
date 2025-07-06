# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

COPY config.yaml .
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]