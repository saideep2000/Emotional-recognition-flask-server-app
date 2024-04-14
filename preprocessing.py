import pandas as pd
from sklearn.model_selection import train_test_split
from deepface import DeepFace


import os
from deepface import DeepFace
import numpy as np
from PIL import Image

def create_deepface_reps(img_file):
    models = ["VGG-Face"]
    try:
        # Save the uploaded file temporarily
        temp_img_path = "temp_image.jpg"  # Consider using a more unique name or a temporary file
        img_file.save(temp_img_path)

        # Pass the file path to DeepFace
        embeddings = DeepFace.represent(img_path=temp_img_path, model_name=models[0])

        # Clean up: remove the temporary file
        os.remove(temp_img_path)

        print()

        return embeddings[0]['embedding']
    except Exception as e:
        print(e)
        return None


def preprocess_data(df):
    # Calculate variance for each column
    variances = df.var()

    # Sort variances in ascending order to get columns with the lowest variance first
    # and then select the first 2048 columns
    # low_variance_columns = variances.nsmallest(2048).index

    low_variance_columns = variances.nsmallest(4096).index

    # Keep only the columns with the lowest variance
    df = df[low_variance_columns]

    return df

