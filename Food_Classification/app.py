import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import os
import requests
import numpy as np
import json
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from google.cloud import storage
import tempfile

import sqlite3

app = Flask(__name__)

# This is set during the Docker image creation
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/samra/zWork with GCP/KEY-FILE.json"


# Create the database using SQLite3 (if it wasn't created in a former run)
def create_db():
    # Check if the file exists on Google Cloud Storage
    bucket_name = "segmented-volume-images"
    blob_name = "shared_db/nutValDB.db"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # ###THERE ARE ERRORS HERE, NEED TO FIX###
    if not blob.exists():
        # Create the table and save the database locally
        db_path = "shared_db/nutValDB.db"
        conn_obj = sqlite3.connect(db_path)
        cursor_obj = conn_obj.cursor()
        table = "CREATE TABLE NUTRITIONAL_VALUES (Name VARCHAR(255) NOT NULL, Serving_Weight_Grams INT, Calories INT," \
                "Total_Fat INT,Cholesterol INT,Sodium INT,Potassium INT,Total_Carbohydrates INT,Sugars INT," \
                "Protein INT);"
        cursor_obj.execute(table)
        print("Successfully created table")
        conn_obj.close()

        # Upload the database file to Google Cloud Storage
        blob.upload_from_filename(db_path)
        print("Successfully uploaded database file to Google Cloud Storage")
        # Remove the temporary local file
        # db_path.close() #this is wrong of course... need to create temp file to close
        # os.remove(db_path)

    else:
        print("The database file already exists on Google Cloud Storage")


# Insert data into db
def insert_into_table(nut_values_tuple):
    db_path = "shared_db/nutValDB.db"

    # Connect to the existing database file on Google Cloud Storage
    bucket_name = "segmented-volume-images"
    blob_name = db_path

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    download_blob = bucket.blob(blob_name)

    # Use the system temporary directory
    temp_dir = tempfile.gettempdir()

    # Download the existing database file from Google Cloud Storage to a temporary local file
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_local_file:
        download_blob.download_to_filename(temp_local_file.name)

        # Connect to the SQLite database
        conn_obj_1 = sqlite3.connect(temp_local_file.name)
        cursor_obj_1 = conn_obj_1.cursor()

        insert_query = "INSERT INTO NUTRITIONAL_VALUES(Name,Serving_Weight_Grams, Calories,Total_Fat,Cholesterol," \
                       "Sodium,Potassium,Total_Carbohydrates,Sugars,Protein) VALUES(?,?,?,?,?,?,?,?,?,?)"
        # val1,  val2,                 val3,    val4,      val5,     val6,  val7,         val8,           val9,  val10
        # insert_query = "DELETE FROM NUTRITIONAL_VALUES WHERE Name LIKE '%macaroons%'" # DELETE  (use only for testing)
        cursor_obj_1.execute(insert_query, nut_values_tuple)
        conn_obj_1.commit()
        conn_obj_1.close()

        upload_blob = bucket.blob(blob_name)
        # Upload the modified database file back to Google Cloud Storage
        upload_blob.upload_from_filename(temp_local_file.name)

        # Remove the temporary local file
        # temp_local_file.close()
        # os.remove(temp_local_file.name)
        print("Successfully inserted data into the table and uploaded the modified db file to Google Cloud Storage")


# View data from db
# This is only used manually to check the tables in the local DB
@app.route('/db', methods=['GET'])
def view_db():
    db_path = "shared_db/nutValDB.db"

    # Connect to the existing database file on Google Cloud Storage
    bucket_name = "segmented-volume-images"
    blob_name = db_path

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Use the system temporary directory
    temp_dir = tempfile.gettempdir()

    # Download the existing database file from Google Cloud Storage to a temporary local file
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_local_file:
        blob.download_to_filename(temp_local_file.name)

        # Connect to the SQLite database
        conn_obj_3 = sqlite3.connect(temp_local_file.name)
        cursor_obj_3 = conn_obj_3.cursor()

        view_query = "SELECT * FROM NUTRITIONAL_VALUES"
        cursor_obj_3.execute(view_query)
        data = cursor_obj_3.fetchall()
        for row in data:
            print(row)
        conn_obj_3.close()

        # Remove the temporary local file
        # temp_local_file.close()
        # os.remove(temp_local_file.name)
        print("Successfully fetched and displayed data from the database file on Google Cloud Storage")
        return jsonify(data)


# NutritionIX API for retrieving nutritional values of identifies food AND saving it in our the local database
# First, check if this class of food was previously identified for its nutritional values
# If it doesn't exist in the local database, use the API and then save the retrieved results to the local database
def nut_values(food_name):
    db_path = "shared_db/nutValDB.db"

    # Connect to the existing database file on Google Cloud Storage
    bucket_name = "segmented-volume-images"
    blob_name = db_path

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Use the system temporary directory
    temp_dir = tempfile.gettempdir()

    # Download the existing database file from Google Cloud Storage to a temporary local file
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_local_file:
        blob.download_to_filename(temp_local_file.name)

        conn_obj_2 = sqlite3.connect(temp_local_file.name)

        print("Successfully fetched nutritional values from the database file on Google Cloud Storage")

        cursor_obj_2 = conn_obj_2.cursor()
        # print(food_name)

        query = "SELECT Name FROM NUTRITIONAL_VALUES WHERE Name=?"
        cursor_obj_2.execute(query, (food_name,))
        # print(cursor_obj.fetchone())

        full_str = cursor_obj_2.fetchone()

        if full_str is not None and food_name in full_str:  # Nutritional value exists in local db
            query = "SELECT * FROM NUTRITIONAL_VALUES WHERE Name=?"
            cursor_obj_2.execute(query, (food_name,))
            nut_tuple = cursor_obj_2.fetchone()
            conn_obj_2.close()
            # Remove the temporary local file
            # temp_local_file.close()
            # os.remove(temp_local_file.name)
            return list(nut_tuple)

        else:  # Need to get nutritional values from API and save them in local db for future results
            end_pt_url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
            HEADERS = {
                "Content-Type": "application/json",
                "x-app-id": "b663319a",
                "x-app-key": "09f02c8d76e65b2570d3d89b3e519b8d"
            }
            query = {
                "query": food_name,
            }
            # print(query)

            r = requests.post(end_pt_url, headers=HEADERS, json=query)
            data = json.loads(r.text)
            list_tuples_data = data.items()
            conv_to_str = ''.join(''.join(map(str, l)) for l in list_tuples_data)
            to_arr = conv_to_str.split(",")  # Full nutritional information as a list

            needed_values = to_arr[4:16]
            # print(needed_values)
            # print("\n")
            # print(to_arr)

            list_values = [food_name]  # Initialize list to add to db

            # print(list_values)
            # print("\n")

            # Build list of wanted values
            # for i in range(0,16):
            #  index = to_arr[i].find(":")
            #  print(to_arr[i])
            # print(to_arr[i][index+2:])

            # Values are fixed from API, so it's possible to just enter them manually into the list
            if "serving_weight_grams" in needed_values[0]:  # Weight in grams
                i = 0
                index = needed_values[i].find(":")
                list_values.append(needed_values[i][index + 2:])
            else:
                i = 1
                index = needed_values[i].find(":")
                list_values.append(needed_values[i][index + 2:])

            i += 1  # Calories
            index = needed_values[i].find(":")
            list_values.append(needed_values[i][index + 2:])

            i += 1  # Total fat
            index = needed_values[i].find(":")
            list_values.append(needed_values[i][index + 2:])

            i += 2  # Cholesterol
            index = needed_values[i].find(":")
            list_values.append(needed_values[i][index + 2:])

            i += 1  # Sodium
            index = needed_values[i].find(":")
            list_values.append(needed_values[i][index + 2:])

            i += 5  # Potassium
            index = needed_values[i].find(":")
            list_values.append(needed_values[i][index + 2:])

            i -= 4  # Total carbohydrates
            index = needed_values[i].find(":")
            list_values.append(needed_values[i][index + 2:])

            i += 2  # Sugars
            index = needed_values[i].find(":")
            list_values.append(needed_values[i][index + 2:])

            i += 1  # Protein
            index = needed_values[i].find(":")
            list_values.append(needed_values[i][index + 2:])

            # print(list_values)

            for k in range(len(list_values)):
                if list_values[k] == "None":
                    list_values[k] = '0'

            # print(list_values)

            # Convert list to tuple
            tuple_values = tuple(list_values)

            # Insert into local db
            # print(tuple_values)
            insert_into_table(tuple_values)

            # Remove the temporary local file
            # temp_local_file.close()
            # os.remove(temp_local_file.name)

            # Return list
            return list_values

            # Name,Serving_Weight_Grams, Calories,Total_Fat,Cholesterol,Sodium,Potassium,Total_Carbohydrates,Sugars,
            # Protein Need to find substring 'serving_weight_grams', 'nf_calories', 'nf_total_fat', 'nf_cholesterol',
            # 'nf_sodium', 'nf_potassium', 'nf_total_carbohydrate', 'nf_sugars', 'nf_protein'


def predict_class(model, images, show=True):
    # Create a GCS client
    storage_client = storage.Client()

    labels_values = []
    for img in images:
        # Load image from GCS
        bucket_name, file_path = img.split('/', 1)
        bucket_name = 'segmented-volume-images'
        bucket = storage_client.bucket(bucket_name)
        file_path = 'segmented_images/' + file_path
        blob = bucket.blob(file_path)

        # Use the system temporary directory
        temp_dir = tempfile.gettempdir()

        # Download the image to a temporary file
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)

            # Load the image using Keras
            img = image.load_img(temp_file.name, target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img /= 255.
            '''
            plt.imshow(img[0])
            plt.axis('off')
            plt.show()
            '''

            predict_img = model.predict(img)
            index = np.argmax(predict_img)
            food_list.sort()
            predict_value = food_list[index]
            # print(predict_value)

            nut_values_list = nut_values(predict_value)  # Get nutritional values list
            # nut_values_list[0] = nut_values_list[0].replace("_", " ").title()
            nut_values_print = "Food Name: " + nut_values_list[0].replace("_", " ").title()
            nut_values_print += "\nServing weight in grams: " + str(nut_values_list[1])
            nut_values_print += "\nAmount Per Serving"
            nut_values_print += "\n\tCalories: " + str(nut_values_list[2])
            nut_values_print += "\n\tTotal Fat: " + str(nut_values_list[3]) + "g"
            nut_values_print += "\n\tCholesterol: " + str(nut_values_list[4]) + "mg"
            nut_values_print += "\n\tSodium: " + str(nut_values_list[5]) + "mg"
            nut_values_print += "\n\tPotassium: " + str(nut_values_list[6]) + "mg"
            nut_values_print += "\n\tTotal Carbohydrates: " + str(nut_values_list[7]) + "g"
            nut_values_print += "\n\tSugars: " + str(nut_values_list[8]) + "g"
            nut_values_print += "\n\tProtein: " + str(nut_values_list[9]) + "g"

            labels_values.append(nut_values_list)

            if show:
                plt.imshow(img[0])
                plt.axis('off')
                plt.title(predict_value.replace("_", " ").title())
                plt.show()
                print(nut_values_print)

            # temp_file.close()
            # os.remove(temp_file.name)
            # Delete the file from GCS
            # blob.delete()

    # Contains results as [#1 label, #1 nut values, #2 label, #2 nut values, etc.]
    return labels_values


@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON input data
    json_data = request.get_json()

    # Extract the filenames from the JSON
    # Make a list of the segmented images
    to_predict = []
    for i in range(0, len(json_data["image_volume_lst"]), 2):
        to_predict.append(json_data["image_volume_lst"][i])
    # print(to_predict)

    # Easily make a list variable of all classes from "classes.txt"
    global food_list
    food_list = open("model/classes.txt").read().splitlines()
    # print(food_list)

    # False so results don't plot
    labels_values_list = predict_class(model_best, to_predict, False)
    # view_db()  # Test if nutritional values were correctly added
    # print('*********************labels+val')
    # print(labels_values_list)

    # Extract the filenames from the JSON, but this time include the volume estimations
    label_segments_values = []
    for i in range(0, len(json_data["image_volume_lst"])):
        label_segments_values.append(json_data["image_volume_lst"][i])
    # print('*********************path+vol in L')
    # print(label_segments_values)

    # Combine everything into one list, [segment_img, volume, label_values]
    index = 2
    for elem in labels_values_list:
        # print('*********************ELEM')
        # print(elem)
        # print('*********************')
        label_segments_values[index-1] *= 1000  # Liter to mL
        tmp = elem[1]
        for val in range(2, len(elem)):
            # print(elem[val])
            # print(tmp)
            elem[val] = (float(elem[val]) / float(tmp)) * label_segments_values[index-1]
        label_segments_values.insert(index, elem)
        index += 3
    # print(label_segments_values)
    # view_db()

    return jsonify({"img_volume_label_nut_val": label_segments_values})


@app.route('/health', methods=['GET'])
def health_check():
    return "Healthy"
	

# Loading the model to make predictions
K.clear_session()
print("* Loading classification model")
# model_path = 'D:/Google Drive/Software Engineering/model_trained_101class.hdf5'
model_path = 'model/model_trained_101class.hdf5'
model_best = load_model(model_path, compile=False)
print("* Classification model is loaded")
# Create DB file (if it wasn't created before)
create_db()

food_list = []

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
