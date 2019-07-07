import os
from flask import Flask, render_template, request,jsonify
from imageio import imread,imwrite
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input as preprocess_img_net
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D,Dropout,Flatten,Activation
from keras.models import Sequential
from sklearn.externals import joblib
import cv2
import tensorflow as tf
import numpy as np
import time
import json
app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
modelface = None
graph = None
modelvgg = None
model = None

def create_face_recog_model():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.model.load_weights('facerecogweights.h5')
    model =Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return model

def create_model():
    modelfacereog=create_face_recog_model()
    filename = 'modelface.sav'
    loaded_model = joblib.load(filename)
    modelvgg = VGG16()
    modelvgg.layers.pop()
    modelvgg = Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    graph = tf.get_default_graph()
    return loaded_model,modelvgg,graph,modelfacereog

def load_mod():
    global modelvgg,modelface,graph,modelfacereog
    modelface,modelvgg,graph,modelfacereog= create_model()

def predict_image(path,target_size=224): 
    image = load_img(path, target_size=(target_size, target_size))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    img_feature = modelvgg.predict(image, verbose=0)
    return img_feature

def crop_img(x,y,w,h,path):
    img = imread(str(path))
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    faceimg = img[ny:ny+nr, nx:nx+nr]
    faceimg =cv2.resize(faceimg,(40,40))
    return faceimg

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_img_net(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def verifyFace(img1, img2):
    epsilon = 0.40
    img1_representation = modelfacereog.predict(preprocess_image(img1))[0,:]
    img2_representation = modelfacereog.predict(preprocess_image(img2))[0,:]
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    
    print("Cosine similarity: ",cosine_similarity)
    print("Euclidean distance: ",euclidean_distance)
    
    if(cosine_similarity < epsilon):
        return True
    else:
        return False

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    with graph.as_default():
        isSpoofed = False
        file = request.files['image']
        name = str(int(time.time())) + str('.jpg')
        f = os.path.join(app.config['UPLOAD_FOLDER'],name)
        file.save(f)
        result = predict_image(f)
        result = modelface.predict(list(result))
        if np.round(result)==1:
            isSpoofed = True
            os.remove(f)
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
            imwrite(f, croppedimg)

            # with open(f, "rb") as imageFile:
            #     base64img = base64.b64encode(imageFile.read())
            #     data2["data"]= base64img.decode('ascii')
            # os.remove(f)
            data2["data"]= "/static/"+name
            return jsonify({"facedata":str(data),"base64img":data2,"isSpoofed":str(isSpoofed)})



@app.route('/verify', methods=['POST','GET'])
def verify_face():
    with graph.as_default():
        img1=request.form['main_image']
        img2=request.form['current_image']
        result = verifyFace(img1,img2)
        return jsonify({"result":result})
        


if __name__ == "__main__":
    print(("* Starting server ..."))
    load_mod()
    app.run(debug=True,host='0.0.0.0',port=5000,threaded=True)