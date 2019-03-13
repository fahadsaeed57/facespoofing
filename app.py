import os
from flask import Flask, render_template, request,jsonify
from scipy.misc import imread, imresize,imsave
import numpy as np
import time
import json
import base64
app = Flask(__name__,static_url_path='')

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
    data2={}
    file = request.files['image']
    name = str(int(time.time())) + str('.jpg')
    f = os.path.join(app.config['UPLOAD_FOLDER'],name)
    file.save(f)
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
    return jsonify({"facedata":str(data),"base64img":data2})


if __name__ == "__main__":
    print(("* Starting server ..."))
    app.run(debug=True)