import pickle
from flask import Flask
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.pipeline import Pipeline

app = Flask(__name__)

@app.route("/")
def home():
    img = get_current_crossing_image()
    img = transform_to_1D(img)
    model = load_model()
    prediction = (str(model.predict([img])))
    return prediction

def get_current_crossing_image():
    image_request = requests.get("http://rrcrossings.woodhavenmi.org/allen.jpg?rnd=")
    image_bytes = BytesIO(image_request.content)
    return Image.open(image_bytes)

def transform_to_1D(img: Image):
    return np.asarray(img).ravel()

def load_model() -> Pipeline:
    with open("./src/model/tot_model", 'rb') as pickle_file:
        return pickle.load(pickle_file)