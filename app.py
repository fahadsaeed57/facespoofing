import os
from flask import Flask, render_template, request,jsonify
from scipy.misc import imread, imresize,imsave
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from sklearn.externals import joblib
import tensorflow as tf
import numpy as np
import time
import json
import base64
app = Flask(__name__,static_url_path='')

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
modelface = None
graph = None
modelvgg = None
def create_model():
    filename = 'modelface.sav'
    loaded_model = joblib.load(filename)
    graph = tf.get_default_graph()
    return loaded_model,graph
def load_mod():
    global modelface,graph
    modelface,graph= create_model()
def predict_image(path,target_size=224): 
    modelvgg = VGG16()
    modelvgg.layers.pop()
    modelvgg = Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    image = load_img(path, target_size=(target_size, target_size))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    img_feature = modelvgg.predict(image, verbose=0)
    return img_feature
def crop_img(x,y,w,h,path):
    img = imread(str(path), mode='RGB')
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    faceimg = img[ny:ny+nr, nx:nx+nr]
    faceimg = imresize(faceimg,(120,120))
    return faceimg

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    isSpoofed = False
    file = request.files['image']
    name = str(int(time.time())) + str('.jpg')
    f = os.path.join(app.config['UPLOAD_FOLDER'],name)
    file.save(f)
    # load_mod()
    with graph.as_default():
        result = predict_image(f)
        result = modelface.predict(list(result))
        if np.round(result)==1:
            isSpoofed = True
        if isSpoofed == True:
            return jsonify({"isSpoofed":str(isSpoofed)})
        if isSpoofed == False:
            data2={}
            data = request.form['data']
            data = json.loads(data)
            x = int(data["x"])
            y = int(data["y"])
            w = int(data["width"])
            h = int(data["height"])
            croppedimg = crop_img(x,y,w,h,f)
            imsave(f, croppedimg)
            with open(f, "rb") as imageFile:
                base64img = base64.b64encode(imageFile.read())
                data2["data"]= base64img.decode('ascii')
            return jsonify({"facedata":str(data),"base64img":data2,"isSpoofed":str(isSpoofed)})


if __name__ == "__main__":
    print(("* Starting server ..."))
    load_mod()
    app.run(debug=True)