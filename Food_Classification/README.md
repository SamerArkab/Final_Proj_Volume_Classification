# Food Classification and Nutrition Value Estimation Application

This application uses deep learning models to classify food items in images and computes their nutritional values. It consists of three main scripts:

1. Training Model Code: This script trains the food classification model using image datasets.
2. Classification Code: This script uses the trained model to classify food items in new, unseen images.
3. Nutrition Value Retrieval and Storage Code: This script retrieves nutritional values of identified food items using the NutritionIX API, and stores these values in a local SQLite database.

## Training Model Code

The training code uses Keras with a TensorFlow backend to train a deep learning model on a dataset of food images. It uses a pre-trained InceptionV3 model with additional dense layers to classify the food items. The script includes code for data augmentation, which helps to improve the model's performance by generating more diverse training samples.

The script assumes you have a folder structure where each category of food has its own folder of images. These should all be placed within a single parent folder. The script uses 80% of the images for training and 20% for validation.

* I used [Multiclass Food Classification using TensorFlow](https://www.kaggle.com/code/theimgclist/multiclass-food-classification-using-tensorflow/notebook) as a guideline for training my model.

## Classification Code

The classification script uses the trained model to classify food items in new images. It also estimates the volume of the food item in the image, which helps to estimate the portion size of the food item.

## Nutrition Value Retrieval and Storage Code

The nutritional value retrieval and storage code retrieves the nutritional values of the identified food items using the NutritionIX API. If the nutritional values of the food item have previously been fetched, the code retrieves these values from a local SQLite database instead of using the API. If the nutritional values have not been fetched before, the script uses the API and then stores the retrieved values in the local database for future use.

The application is hosted on a server and provides a REST API for clients to use. Clients send a POST request with the images they wish to classify, and the server returns the classified food items and their nutritional values.

This application is a prototype and is intended for educational and demonstrative purposes only. It should not be used for production or for making health-related decisions without further validation and testing.
