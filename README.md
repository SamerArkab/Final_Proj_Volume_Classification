* To extend the utility and accessibility of the project, we've also developed a client-server based application which runs the machine learning models. This application is built with Node.js and React for a robust and responsive user experience. Furthermore, the models have been containerized using Docker and deployed to the Google Cloud Platform (GCP), allowing for easy scalability and maintenance.
* [Server-side repository](https://github.com/SamerArkab/volume_classification_server)
* [Client-side repository](https://github.com/SamerArkab/volume_classification_client)

# Food Volume Estimation and Nutrition Value Analysis Application

This application is designed to estimate the volume of food items in images, classify the food items, and provide their nutritional values. It achieves this by integrating a volume estimation model, a classification model, and a nutritional value retrieval and storage system.

The application consists of four main components:

1. Volume Estimation Model: This model estimates the volume of the food items in the images.
2. Training Model Code: This script trains the food classification model using image datasets.
3. Classification Code: This script uses the trained model to classify food items in new, unseen images.
4. Nutrition Value Retrieval and Storage Code: This script retrieves nutritional values of identified food items using the NutritionIX API, and stores these values in a local SQLite database.

## Volume Estimation Model

The volume estimation model estimates the volume of food items in images. It is trained on a dataset of food images with known volumes, and it estimates the volume of new, unseen food items by analyzing their images.

The model uses 3D images of food items to generate more accurate volume estimations. The volume estimations are used in combination with the classification model and nutritional value retrieval system to provide a complete analysis of the food items in the images.

## Training Model Code

The training code uses Keras with a TensorFlow backend to train a deep learning model on a dataset of food images. It uses a pre-trained Xception model with additional dense layers to classify the food items. The script includes code for data augmentation, which helps to improve the model's performance by generating more diverse training samples.

The script assumes you have a folder structure where each category of food has its own folder of images. These should all be placed within a single parent folder. The script uses 80% of the images for training and 20% for validation.

## Classification Code

The classification script uses the trained model to classify food items in new images. The classifications are used in combination with the volume estimations to provide a more detailed nutritional analysis of the food items.

## Nutrition Value Retrieval and Storage Code

The nutritional value retrieval and storage code retrieves the nutritional values of the identified food items using the NutritionIX API. If the nutritional values of the food item have previously been fetched, the code retrieves these values from a local SQLite database instead of using the API. If the nutritional values have not been fetched before, the script uses the API and then stores the retrieved values in the local database for future use.

The application is hosted on a server and provides a REST API for clients to use. Clients send a POST request with the images they wish to analyze, and the server returns the classified food items, their estimated volumes, and their nutritional values.

This application is a prototype and is intended for educational and demonstrative purposes only. It should not be used for production or for making health-related decisions without further validation and testing.
