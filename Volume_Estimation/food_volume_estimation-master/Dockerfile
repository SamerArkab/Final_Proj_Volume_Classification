# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the code from the current directory into the container at /app
COPY . /app

# Add model files to image
COPY models/fine_tune_food_videos/monovideo_fine_tune_food_videos.json models/depth_architecture.json
COPY models/fine_tune_food_videos/monovideo_fine_tune_food_videos.h5 models/depth_weights.h5
COPY models/segmentation/mask_rcnn_food_segmentation.h5 models/segmentation_weights.h5

# Copy the service account key file into the container
COPY KEY-FILE.json /app/keyfile.json

# Copy the setup.py and requirements.txt files into the container at /app
COPY ./food_volume_estimation/setup.py ./food_volume_estimation/requirements.txt /app/

# Install system dependencies
RUN apt update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install food-volume-estimation package
COPY food_volume_estimation/ food_volume_estimation/
RUN python setup.py install

# Copy the additional folder into the container
COPY ./keras/engine/saving.py /usr/local/lib/python3.6/site-packages/keras/engine/

# Set the environment variable for the service account key file
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/keyfile.json

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Set environment variables
ENV MODEL_NAME final_proj_py36

# Run the command to start the model server
CMD ["python", "app.py"]
