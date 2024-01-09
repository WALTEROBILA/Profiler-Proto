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
    return render_template('home.html')

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

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html", prediction_text="The Player's midifeld profile key is {}".format(output))



if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


