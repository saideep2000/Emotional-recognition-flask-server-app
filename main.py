# main.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from preprocessing import create_deepface_reps, preprocess_data
from svm import load_model, predict

app = Flask(__name__)


emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

@app.route("/getemotion-svm", methods=["POST"])
def svm():
    model = load_model('models/DeepFaceModel_SVM.pkl')
    img_path = request.files['image']
    embeddings = create_deepface_reps(img_path)
    if embeddings is None:
        return jsonify({"error": "Failed to process image"}), 400

    # Convert embeddings to DataFrame for preprocessing
    df = pd.DataFrame([embeddings])
    df = preprocess_data(df)
    y_pred = predict(model, df)
    print(y_pred)
    # return jsonify({"prediction": emotions[int(y_pred)]})
    return jsonify({"prediction": y_pred[0]})


if __name__ == '__main__':
    app.run(debug=True)
