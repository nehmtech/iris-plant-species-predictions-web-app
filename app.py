from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the model
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
# Load the model
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:       
        
        # Get values 
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width']),
        ]
        
        # Transform features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Get prediction probability        
        proba = model.predict_proba(features_scaled)
        max_proba = f"{round(np.max(proba) * 100, 2)}%"
        
        
        return jsonify({
            'prediction': prediction[0],
            'probability': max_proba
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__== '__main__':
    app.run(debug=True)
    
