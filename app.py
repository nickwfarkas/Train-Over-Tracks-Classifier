from flask import Flask
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

@app.route("/")
def home():
    img = get_current_crossing_image()
    img = transform_to_1D(img)

    model = load_model()
    try:
        prediction = (str(model.predict([img])))
    except:
        return ("Failed to predict")
    return prediction

def get_current_crossing_image():
    image_request = requests.get("http://rrcrossings.woodhavenmi.org/allen.jpg?rnd=")
    image_bytes = BytesIO(image_request.content)
    return Image.open(image_bytes)

def transform_to_1D(img: Image):
    img = img.convert("L")
    return np.asarray(img).ravel()

# def reduce_image(img: Image):
#     img_arr = transform_to_1D(img)
#     # with open("./src/model/transform.joblib", 'rb') as pca_file:
#     #     pca = load(pca_file)
#     pca = PCA(n_components=60)
#     return pca.fit_transform(img_arr)

# def load_model():
#     with open("./model/model.joblib", 'rb') as model_file:
#         return load(model_file)

def load_model():
    with open("./src/model/tot_model", 'rb') as model_file:
        return pickle.load(model_file)
    
# def load_model():
#     with open("./src/model/tot_model", 'rb') as pickle_file:
#         return pickle.load(pickle_file)