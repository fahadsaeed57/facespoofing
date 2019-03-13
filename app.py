import os
from flask import Flask, render_template, request,jsonify
from scipy.misc import imread, imresize
import numpy as np
import time
app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    file = request.files['image']
    name = str(int(time.time()))
    f = os.path.join(app.config['UPLOAD_FOLDER'],name)
    file.save(f)
    data = request.form['data']
    return jsonify({"message":"uploaded img is " +str(name)"data":str(data)})


if __name__ == "__main__":
    print(("* Starting server ..."))
    app.run(debug=True)