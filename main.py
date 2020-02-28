from flask import Flask,render_template,request,url_for,jsonify

import numpy as np


from keras.models import load_model
import cv2
import requests
import json

import tensorflow as tf


app = Flask(__name__)


@app.route('/<path:url>')
def predict(url):
    
    requesturl = requests.get(url, auth=('user', 'pass'))
    urljson = requesturl.json()
    token = urljson['downloadTokens']

    urlimage = url+"?alt=media&token="+token

    requestimage = requests.get(urlimage,stream = True).raw
    image = np.asarray(bytearray(requestimage.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image*1./255, (299,299), interpolation=cv2.INTER_CUBIC)
    image = image.reshape(-1,299,299,3)
    
      
    # MODEL = load_model('./inception_best_model_AddPic.h5')
    with graph.as_default():
        predict_1 = MODEL.predict(image)

    predictlist = predict_1.tolist()

    sort_index = np.argsort(predictlist)
    sortlist = sort_index.tolist()

    snake_name = ['Cobra','King Cobra','Banded Krait','Malayan Krait','Malayan Pitviper','White lipped Pitviper','Russell Viper']

    result = {
        "species1" : snake_name[sortlist[0][6]],
        "species2" : snake_name[sortlist[0][5]],
        "species3" : snake_name[sortlist[0][4]],
        "result1" : predictlist[0][sortlist[0][6]],
        "result2" : predictlist[0][sortlist[0][5]],
        "result3" : predictlist[0][sortlist[0][4]]
    }
    
    '''
    result = {
            "result1": 0.9361255764961243, 
            "result2": 0.06063055619597435, 
            "result3": 0.0011159885907545686, 
            "species1": "Banded Krait", 
            "species2": "Cobra", 
            "species3": "Russell Viper"
    }'''
    #return jsonify(predictlist)
    return result


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    MODEL = load_model('./inception_best_model_AddPic.h5')
    graph = tf.get_default_graph()
    app.run(host='127.0.0.1', port=8080, debug=True)
    #app.run(host='0.0.0.0', port=80, debug=True)