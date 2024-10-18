from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=["POST"])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Input Validation
    try:
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)
    except ValueError:
        return "Invalid input! Please ensure all fields are filled correctly with numeric values.", 400

    # Prepare DataFrame for prediction
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Predict using the model
    prediction = pipe.predict(input_df)[0] * 1e5
    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
