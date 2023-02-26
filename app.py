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
        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
        output = linear_reg_model.predict(new_data)
        return jsonify(output[0])
    else:
        return "Content type is not supported provide data in JSON format."

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    output = linear_reg_model.predict(final_input)[0]
    return render_template("index.html", prediction_text="The Price for this house is {0}".format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080,  debug=False)