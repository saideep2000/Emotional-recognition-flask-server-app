import pickle
import numpy as np

def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model

def predict(model, X):
    y_pred = model.predict(X)
    return y_pred
