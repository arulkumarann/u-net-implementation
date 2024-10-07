# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run your app
CMD ["python", "app.py"]
