from flask import Flask, render_template, request
from flask import request
import numpy as np 
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

import pickle#Initialize the flask App

app = Flask(__name__, template_folder='templates')

#model = DBSCAN()

filename = 'dbscan_model'
model = pickle.load(open(filename, 'rb'))


@app.route("/")
def index():
    
    return render_template('index.html')


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) 
    return render_template('index.html', prediction_text='CO2    Emission of the vehicle is :{}'.format(output))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)