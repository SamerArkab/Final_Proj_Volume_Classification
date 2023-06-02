import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import os
import requests
import numpy as np
import json
import matplotlib.pyplot as plt

import sqlite3


# Create the database using SQLite3 (if it wasn't created in a former run)
def create_db():
    if not os.path.isfile("localDB/nutValDB.db"):
        conn_obj = sqlite3.connect("localDB/nutValDB.db")  # Returns Connection obj that represent the db
        cursor_obj = conn_obj.cursor()
        table = "CREATE TABLE NUTRITIONAL_VALUES (Name VARCHAR(255) NOT NULL, Serving_Weight_Grams INT, Calories INT," \
                "Total_Fat INT,Cholesterol INT,Sodium INT,Potassium INT,Total_Carbohydrates INT,Sugars INT," \
                "Protein INT);"
        cursor_obj.execute(table)
        print("Successfully created table")
        conn_obj.close()


# Insert data into db
def insert_into_table(nut_values_tuple):
    conn_obj_1 = sqlite3.connect("localDB/nutValDB.db")
    cursor_obj_1 = conn_obj_1.cursor()

    insert_query = "INSERT INTO NUTRITIONAL_VALUES(Name,Serving_Weight_Grams, Calories,Total_Fat,Cholesterol,Sodium," \
                   "Potassium,Total_Carbohydrates,Sugars,Protein) VALUES(?,?,?,?,?,?,?,?,?,?)"
    # val1,  val2,                 val3,    val4,      val5,     val6,  val7,         val8,           val9,  val10
    # insert_query = "DELETE FROM NUTRITIONAL_VALUES WHERE Name LIKE '%macaroons%'" # DELETE  (use only for testing...)
    cursor_obj_1.execute(insert_query, nut_values_tuple)
    conn_obj_1.commit()
    conn_obj_1.close()


# View data from db
# This is only used manually to check the tables in the local DB
def view_db():
    conn_obj_3 = sqlite3.connect("localDB/nutValDB.db")
    cursor_obj_3 = conn_obj_3.cursor()
    view_query = "SELECT * FROM NUTRITIONAL_VALUES"
    cursor_obj_3.execute(view_query)
    print(cursor_obj_3.fetchall())
    conn_obj_3.close()


# NutritionIX API for retrieving nutritional values of identifies food AND saving it in our the local database
# First, check if this class of food was previously identified for its nutritional values
# If it doesn't exist in the local database, use the API and then save the retrieved results to the local database
def nut_values(food_name):
    conn_obj_2 = sqlite3.connect("localDB/nutValDB.db")
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

        # Return list
        return list_values

        # Name,Serving_Weight_Grams, Calories,Total_Fat,Cholesterol,Sodium,Potassium,Total_Carbohydrates,Sugars,
        # Protein Need to find substring 'serving_weight_grams', 'nf_calories', 'nf_total_fat', 'nf_cholesterol',
        # 'nf_sodium', 'nf_potassium', 'nf_total_carbohydrate', 'nf_sugars', 'nf_protein'


def predict_class(model, images, show=True):
    for img in images:
        img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        predict = model.predict(img)
        index = np.argmax(predict)
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

        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(predict_value.replace("_", " ").title())
            plt.show()
            print(nut_values_print)


# Loading the model to make predictions
K.clear_session()
model_best = load_model('model/model_trained_101class.hdf5', compile=False)
# Create DB file (if it wasn't created before)
create_db()
# Easily make a list variable of all classes from "classes.txt"
food_list = open("model/classes.txt").read().splitlines()
# print(food_list)

# Make a list of downloaded images and test the trained model - *** EXAMPLE ONLY ***
to_predict = ['test_images/ff.jpg', 'test_images/fr.jpg', 'test_images/s.jpg']

# False so results don't plot
predict_class(model_best, to_predict, False)
# view_db()  # Test if nutritional values were correctly added
