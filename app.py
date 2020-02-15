from os import environ

from flask import Flask, jsonify, request, make_response, abort

from meta import CURRENT_REVISION
from model import predict_image, Net, read_img_from_base64

app = Flask(__name__)

DEBUG = environ.get('DEBUG')


@app.route('/')
def index():
    return jsonify({'data': "Hello, World!"})


@app.route('/meta')
def meta():
    return jsonify({
        'revision': CURRENT_REVISION
    })


@app.route('/predict', methods=['POST'])
def predict():
    model = Net()
    img = read_img_from_base64(request.json['img'])
    pred = predict_image(model, img)
    return jsonify({'prediction': pred})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=DEBUG)
