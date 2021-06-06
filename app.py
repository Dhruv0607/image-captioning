from flask import Flask, json, request, jsonify
import image_caption_train as impt
## Importing the required libraries
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.compat.v1.keras.backend import set_session
#from keras.backend.tensorflow_backend import set_session
import sys, time, os, warnings 
import numpy as np
import pandas as pd 
from collections import Counter
import pickle
from keras.preprocessing.image import load_img, img_to_array
from IPython.display import display
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from collections import OrderedDict

from tensorflow.keras.applications import VGG16

modelvgg = VGG16(include_top=True,weights=None)
## load the locally saved weights 
modelvgg.load_weights("Data/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)

@app.route("/test", methods=['GET'])

def response():
    rst = {"response": "check working good??"}

    return jsonify(rst), 200


@app.route("/filePath", methods=['POST'])

def check():

    modelvgg = VGG16(include_top=True,weights=None)
    ## load the locally saved weights 
    modelvgg.load_weights("Data/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    imgs = OrderedDict()
    npix = 224
    target_size = (npix,npix,3)
    filename = 'E:/PROJECTS/IMages/dog.jpg'
    img = load_img(filename, target_size=target_size)
    img = img_to_array(img)
    nimg = preprocess_input(img)

    y_pred = modelvgg.predict(nimg.reshape( (1,) + nimg.shape[:3]))
    pImgs = y_pred.flatten()


    r = model.predict_caption(pImgs.reshape(1,len(pImgs)))
    rst = {"message" : r}

    return jsonify(rst), 200



if __name__ == "__main__":
    impt.train_model()
    app.run(host="0.0.0.0", port=5000)