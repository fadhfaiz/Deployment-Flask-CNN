from __future__ import division, print_function

# coding=utf-8
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__, template_folder="templates")

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_skripsi.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
model.make_predict_function()

print('\n Model loaded. Start serving on http://127.0.0.1:5000/ \n')

def model_predict(img_path, model):
    img = image.load_img(img_path, color_mode="grayscale", target_size=(448, 448))

    # Preprocessing the image
    x = image.img_to_array(img)
    print(x.shape)
    x = np.true_divide(x, 255)
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    print(x.shape[1:])
    preds = model.predict(x)
    print('Hasil Prediksi', preds)
    print('\n')
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='tf')
    # preds = model.predict(x)
    # print('Hasil Prediksi', preds)

    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        # print(preds)
        pred_class = preds.argmax(axis=1)

        if pred_class == 0:
            result = 'Kulit Domba'
        elif pred_class == 1:
            result = 'Kulit Imitasi'
        elif pred_class == 2:
            result = 'Kulit Kambing'
        else:
            result = 'Kulit Sapi'
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)