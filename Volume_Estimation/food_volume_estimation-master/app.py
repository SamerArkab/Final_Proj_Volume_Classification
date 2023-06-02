# Single-image food volume estimation
# Using a  monocular depth estimation network and a segmentation network,
# we will estimate the volume of the food displayed in the input image.

from flask import Flask, request, jsonify
from keras.models import Model, model_from_json
from food_volume_estimation.volume_estimator import VolumeEstimator
from food_volume_estimation.depth_estimation.custom_modules import *
from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
import numpy as np
from google.cloud import storage
import os

app = Flask(__name__)

# This is set during the Docker image creation
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/samra/zWork with GCP/KEY-FILE.json"

# Set the Google Cloud Storage bucket name
bucket_name = 'segmented-volume-images'

# Set the path within the bucket to store the segmented images
segmented_images_folder = 'segmented_images/'

# Initialize the Google Cloud Storage client
storage_client = storage.Client()

# Ensure the bucket exists, create it if needed
bucket = storage_client.bucket(bucket_name)
if not bucket.exists():
    bucket.create()

global graph
graph = tf.Graph()
with graph.as_default():
    global sess
    sess = tf.Session()
    with sess.as_default():
        # Paths to model architecture/weights
        depth_model_architecture = 'models/fine_tune_food_videos/monovideo_fine_tune_food_videos.json'
        depth_model_weights = 'models/fine_tune_food_videos/monovideo_fine_tune_food_videos.h5'
        segmentation_model_weights = 'models/segmentation/mask_rcnn_food_segmentation.h5'

        # Create estimator object and initialize
        estimator = VolumeEstimator(arg_init=False)
        with open(depth_model_architecture, 'r') as read_file:
            custom_losses = Losses()
            objs = {'ProjectionLayer': ProjectionLayer,
                    'ReflectionPadding2D': ReflectionPadding2D,
                    'InverseDepthNormalization': InverseDepthNormalization,
                    'AugmentationLayer': AugmentationLayer,
                    'compute_source_loss': custom_losses.compute_source_loss}
            model_architecture_json = json.load(read_file)
            estimator.monovideo = model_from_json(model_architecture_json, custom_objects=objs)
        estimator._VolumeEstimator__set_weights_trainable(estimator.monovideo, False)
        estimator.monovideo.load_weights(depth_model_weights)
        estimator.model_input_shape = estimator.monovideo.inputs[0].shape.as_list()[1:]
        depth_net = estimator.monovideo.get_layer('depth_net')
        estimator.depth_model = Model(inputs=depth_net.inputs, outputs=depth_net.outputs, name='depth_model')
        print('[*] Loaded depth estimation model.')
        # Depth model configuration
        MIN_DEPTH = 0.01
        MAX_DEPTH = 10
        estimator.min_disp = 1 / MAX_DEPTH
        estimator.max_disp = 1 / MIN_DEPTH
        estimator.gt_depth_scale = 0.35  # Ground truth expected median depth
        # Create segmentator object
        estimator.segmentator = FoodSegmentator(segmentation_model_weights)
        print('[*] Loaded segmentation model.')
        # Set plate adjustment relaxation parameter
        estimator.relax_param = 0.01


#  input_image = 'test_images/mix.jpg'
@app.route('/estimate_volume', methods=['POST'])
def estimate_volume():
    # Check if the image file is in the request
    if 'img' not in request.files:
        return jsonify({'error': 'No input image provided'})

    # Read the image file from the request
    image_file = request.files['img']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with graph.as_default():
        with sess.as_default():
            # Perform food segmentation
            results = estimator.segmentator.model.detect([image_rgb], verbose=0)
            r = results[0]

            image_volume_lst = []

            # Iterate over segmented masks and estimate volume for each one
            for i in range(r['masks'].shape[2]):
                mask = r['masks'][:, :, i]
                segmented_image = image * mask[:, :, np.newaxis]  # Maintain the original coloring over the mask

                # Save segmented image as a JPEG file
                output_file_name = 'segmented_{}.jpg'.format(i)
                cv2.imwrite(output_file_name, segmented_image)

                # Estimate volume for the segmented image
                plate_diameter = 0  # Set as 0 to ignore plate detection and scaling
                volume, estimated_volume = estimator.estimate_volume(output_file_name,
                                                                     fov=70, plate_diameter_prior=plate_diameter,
                                                                     plot_results=True)

                # Overwrite the segmented image to show only the mask
                masked_image = image.copy()
                masked_image[np.logical_not(mask)] = [255, 255, 255]

                coordinates = np.argwhere(mask)
                x_min, y_min = coordinates.min(axis=0)
                x_max, y_max = coordinates.max(axis=0)

                cropped_image = masked_image[x_min:x_max, y_min:y_max]
                resized_image = cv2.resize(cropped_image, (mask.shape[1], mask.shape[0]))

                output_file_name = 'segmented_{}.jpg'.format(i)
                cv2.imwrite(output_file_name, resized_image)

                # Upload segmented image to Google Cloud Storage
                destination_blob_name = segmented_images_folder + f'segmented_{i}.jpg'
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(output_file_name)

                # Remove the temporary file
                os.remove(output_file_name)

                image_volume_lst.append(destination_blob_name)
                image_volume_lst.append(estimated_volume)

            # Return results as a JSON response, as follows (key which maps to a list):
            '''
            {
                "image_volume_lst": [
                   "segmented_0.jpg",
                    0.02611716884676605,
                    "segmented_1.jpg",
                    0.5526645696113113,
                    "segmented_2.jpg",
                    0.3282629351905965,
                    "segmented_3.jpg",
                    0.010706482048327626
                ]
            }
            '''
            return jsonify({"image_volume_lst": image_volume_lst})


@app.route('/health', methods=['GET'])
def health_check():
    return "Healthy"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')
