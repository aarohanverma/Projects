# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code to the container
COPY . /app

# Expose the port used by Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.enableCORS", "false"]

