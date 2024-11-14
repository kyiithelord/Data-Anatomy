from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('models/train_model.pkl')

@app.route('/')
def home():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        income = float(request.form['income'])
        location = request.form['location']
        
        # Encode location (Urban = 1, Rural = 0)
        location_map = {'Urban': 1, 'Rural': 0}
        location_encoded = location_map.get(location, 0)  # Default to 0 (Rural) if invalid
        
        # Prepare the input for the model
        features = np.array([[age, income, location_encoded]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Convert the prediction to a native Python int (from int64)
        prediction = int(prediction)

        # Return the result as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
