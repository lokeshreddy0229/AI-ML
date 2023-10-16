from flask import Flask, render_template, request
import pickle
import numpy as np
import pickle
import pandas as pd


app = Flask(__name__)

# Load the pre-trained model from the pickle file
model=pickle.load(open(r"thyroid_1_model.pkl",'rb'))
le5=pickle.load(open(r"label_encoder.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        bp = float(request.form['bp'])
        cholesterol = float(request.form['cholesterol'])
        na_to_k = float(request.form['na_to_k'])

        # Prepare input data for prediction
        input_data = [[bp, cholesterol, na_to_k]]

        # Make a prediction using the loaded model
        prediction = model.predict(input_data)
        prediction=le5.inverse_transform(prediction)
        print(prediction[0])
        if prediction[0]==0:
            result = "Drug Y has to be taken"
        elif prediction[0]==1:
            result = "Drug A has to be taken"
        elif prediction[0]==2:
            result = "Drug B has to be taken"
        elif prediction[0]==3:
            result = "Drug C has to be taken"
        else:
            result = "Drug X has to be taken"            

        return render_template('prediction.html', result=result)

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
