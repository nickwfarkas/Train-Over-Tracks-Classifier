import pickle
from flask import Flask
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.pipeline import Pipeline
from joblib import dump, load

app = Flask(__name__)

@app.route("/")
def home():
    img = get_current_crossing_image()
    img = transform_to_1D(img)
    model = load_model()
    return ("Completed 2")
    prediction = (str(model.predict([img])))
    print("Completed 3")

    return prediction

def get_current_crossing_image():
    image_request = requests.get("http://rrcrossings.woodhavenmi.org/allen.jpg?rnd=")
    image_bytes = BytesIO(image_request.content)
    return Image.open(image_bytes)

def transform_to_1D(img: Image):
    return np.asarray(img).ravel()

def load_model():
    with open("./src/model/model.joblib", 'rb') as model_file:
        return load(model_file)

# def load_model():
#     with open("./src/model/tot_model", 'rb') as pickle_file:
#         return pickle.load(pickle_file)