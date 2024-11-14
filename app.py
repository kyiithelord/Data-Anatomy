from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure the path is correct)
try:
    model = joblib.load('models/train_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    # Render the main page with no prediction initially
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Error: Model not found")

    try:
        # Get form data (convert inputs to float)
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])

        # Prepare input for the model (ensure it's a 2D array as expected by most models)
        features = np.array([[feature1, feature2, feature3]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Return prediction result to the template
        return render_template('index.html', prediction=f"Prediction: {prediction}")

    except Exception as e:
        # Handle any errors in input conversion or prediction
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
