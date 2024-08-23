import tensorflow as tf
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('./saved_model/model.plk', 'rb'))

# Define the percentile for anomaly detection
PERCENTILE = 95  # Adjust this percentile based on your requirements

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.json
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Calculate the threshold based on the specified percentile
        threshold = np.percentile(predictions, PERCENTILE)
        
        # Apply the threshold to determine anomalies
        anomalies = (predictions > threshold).astype(int)
        
        # Return predictions and anomalies as JSON
        return jsonify({
            'predictions': predictions.tolist(),
            'anomalies': anomalies.tolist(),
            'threshold': threshold
        })
    except Exception as e:
        # Handle exceptions and return an error message
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
