import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
## Load the model from the pickle file
linear_reg_model = pickle.load(open('./model_file/linear_regmodel.pkl', 'rb'))
scaler = pickle.load(open('./model_file/scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = request.json["data"]
        print(data) # debug print
        print(np.array(list(data.values())).reshape(1, -1)) # debug print
        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
        output = linear_reg_model.predict(new_data)
        print(output[0]) # debug print
        return jsonify(output[0])
    else:
        return "Content type is not supported."

if __name__ == "__main__":
    app.run(debug=True)