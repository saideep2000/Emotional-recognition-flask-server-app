from flask import Blueprint, request, jsonify
import pandas as pd
import requests
from io import BytesIO
from .preprocessing import create_deepface_reps, preprocess_data
from .svm import predict
import joblib

main = Blueprint('main', __name__)
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

model_url = 'https://assests-for-emotion-recognition-flask-app.s3.amazonaws.com/DeepFaceModel_SVM.pkl'

def load_model_from_url(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    with BytesIO() as model_file:
        for chunk in response.iter_content(chunk_size=8192):
            model_file.write(chunk)
        model_file.seek(0)
        model = joblib.load(model_file)
    return model

@main.route("/getemotion-svm", methods=["POST"])
def svm():
    model = load_model_from_url(model_url)
    img_path = request.files['image']
    embeddings = create_deepface_reps(img_path)
    if embeddings is None:
        return jsonify({"error": "Failed to process image"}), 400

    # Convert embeddings to DataFrame for preprocessing
    df = pd.DataFrame([embeddings])
    df = preprocess_data(df)
    y_pred = predict(model, df)
    return jsonify({"prediction": emotions[int(y_pred)]})
