import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
##Load the model
model=pickle.load(open('svmpoly.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)

    output_value = int(output[0]) #Converting the int64 value to a regular python integer, to avoid TypeError: Object of type int64 is not JSON serializable
   
    print(output_value)
    return jsonify(output_value)

if __name__=='__main__':
    app.run(debug=True)


