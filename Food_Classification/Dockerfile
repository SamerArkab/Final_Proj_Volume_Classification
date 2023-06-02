# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the code from the current directory into the container at /app
COPY . /app

# Set the environment variable for the service account key file
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/KEY-FILE.json

# Make port 8081 available to the world outside this container
EXPOSE 8081

# Set environment variables
ENV MODEL_NAME final_proj_py310

# Run the command to start the model server
CMD ["python", "app.py"]
