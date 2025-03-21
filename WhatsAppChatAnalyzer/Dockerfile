# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building some packages
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port that Streamlit uses (default is 8501)
EXPOSE 8501

# Command to run the app using Streamlit
CMD ["streamlit", "run", "app.py", "--server.enableCORS", "false"]

