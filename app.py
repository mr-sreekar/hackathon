"""
ML Hackathon Project: Real-Time House Price Predictor
====================================================
A Flask web application running a Multiple Linear Regression model 
built completely from scratch using NumPy.

Author: Antigravity AI Assistant
Date: June 2026
"""

import os
from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

# --- Multiple Linear Regression from Scratch ---
class MultipleLinearRegression:
    def __init__(self, lr=0.1, epochs=150):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        
        # Scaling parameters
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # Min-Max Scaling (normalizes features to range [0, 1])
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        # Handle zero-variance features
        range_X = self.X_max - self.X_min
        range_X[range_X == 0] = 1.0
        X_scaled = (X - self.X_min) / range_X
        
        self.y_min = y.min()
        self.y_max = y.max()
        range_y = self.y_max - self.y_min
        if range_y == 0:
            range_y = 1.0
        y_scaled = (y - self.y_min) / range_y

        # Gradient Descent Loop
        for epoch in range(self.epochs):
            # Forward pass: y = Xw + b
            y_pred = np.dot(X_scaled, self.weights) + self.bias
            
            # Loss: Mean Squared Error (MSE)
            loss = np.mean((y_pred - y_scaled) ** 2)
            self.loss_history.append(float(loss))
            
            # Backward pass: Compute gradients
            dw = (2 / n_samples) * np.dot(X_scaled.T, (y_pred - y_scaled))
            db = (2 / n_samples) * np.sum(y_pred - y_scaled)
            
            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        range_X = self.X_max - self.X_min
        range_X[range_X == 0] = 1.0
        X_scaled = (X - self.X_min) / range_X
        
        # Scale back the prediction to original y range
        y_pred_scaled = np.dot(X_scaled, self.weights) + self.bias
        y_pred = y_pred_scaled * (self.y_max - self.y_min) + self.y_min
        return y_pred


# --- Data Store (In-Memory) ---
# Features: [Size (sqft), Bedrooms, Age (years)]
initial_X = np.array([
    [850, 2, 5], [1200, 3, 8], [1500, 3, 15], [1800, 4, 3], [2200, 4, 12],
    [900, 1, 10], [1100, 2, 20], [2500, 5, 2], [3000, 5, 6], [1600, 3, 18],
    [700, 1, 30], [1300, 3, 12], [1400, 2, 4], [2100, 4, 9], [2800, 5, 1],
    [1000, 2, 11], [1750, 3, 7], [1900, 4, 14], [2300, 4, 5], [3200, 5, 10]
], dtype=float)

# Target: Price in thousands of dollars ($k)
initial_y = np.array([
    230, 310, 350, 450, 490,
    190, 210, 610, 680, 340,
    120, 290, 360, 470, 690,
    220, 395, 410, 530, 660
], dtype=float)

dataset_X = initial_X.copy()
dataset_y = initial_y.copy()

# Initialize and train ML model
model = MultipleLinearRegression(lr=0.1, epochs=200)
model.fit(dataset_X, dataset_y)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        size = float(data['size'])
        bedrooms = float(data['bedrooms'])
        age = float(data['age'])
        
        X_new = np.array([[size, bedrooms, age]])
        pred_price = model.predict(X_new)[0]
        
        # Round price to nearest dollar
        return jsonify({
            'success': True,
            'prediction': round(float(pred_price), 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data', methods=['GET'])
def get_data():
    # Convert numpy arrays to list for JSON response
    return jsonify({
        'X': dataset_X.tolist(),
        'y': dataset_y.tolist(),
        'loss_history': model.loss_history,
        'weights': model.weights.tolist(),
        'bias': float(model.bias)
    })


@app.route('/api/train', methods=['POST'])
def train_model():
    global dataset_X, dataset_y
    try:
        data = request.json
        size = float(data['size'])
        bedrooms = float(data['bedrooms'])
        age = float(data['age'])
        price = float(data['price'])
        
        # Add new point to dataset
        new_point_X = np.array([[size, bedrooms, age]])
        new_point_y = np.array([price])
        
        dataset_X = np.vstack([dataset_X, new_point_X])
        dataset_y = np.append(dataset_y, new_point_y)
        
        # Retrain the model on the expanded dataset
        model.fit(dataset_X, dataset_y)
        
        return jsonify({
            'success': True,
            'dataset_size': len(dataset_y),
            'loss_history': model.loss_history
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("Starting ML Web App on http://127.0.0.1:5000")
    app.run(debug=True)
