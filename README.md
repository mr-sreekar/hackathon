# 🚀 Real-Time ML House Price Predictor - Hackathon Dashboard

An interactive Web-based Machine Learning application built for the hackathon. It features a custom **Multiple Linear Regression (MLR) model built from scratch using NumPy** on the backend (Flask), and a **sleek dark-theme dashboard frontend** using TailwindCSS, Lucide Icons, and Chart.js.

---

## 🎨 Application Interface

The dashboard includes:
1. **Interactive Price Inference**: Key in property Size (Sq Ft), Bedrooms, and Property Age to get an instant price estimation calculated using custom gradient descent weight matrices.
2. **On-the-Fly Model Training**: Add new custom properties (features and actual selling prices) to the database and trigger real-time gradient descent training. The backend model immediately updates and pushes new coefficients and loss statistics back to the frontend.
3. **Real-Time Visualization Panels**:
   - **Size vs Price & Regression Fit**: Renders property size vs price scatter plot alongside the dynamically updated model regression plane.
   - **Training Loss (MSE)**: Renders the gradient descent convergence loss curve (Mean Squared Error) showing training optimization over epochs.

---

## 🛠️ Tech Stack & Architecture

- **Backend**: Python, Flask, NumPy (Custom Multiple Linear Regression from scratch with min-max scaling).
- **Frontend**: HTML5, Vanilla JavaScript, CSS3, Tailwind CSS (Design System), Lucide Icons, Chart.js (Real-time charts).
- **No Heavy ML Libraries**: The ML engine is custom-coded (no `scikit-learn` or `TensorFlow`), making it lightweight, transparent, and easy to explain.

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python installed. Then, install Flask and NumPy:
```bash
pip install Flask numpy
```

### 2. Run the Server
From the project directory, execute:
```bash
python app.py
```

### 3. Open the Dashboard
Navigate to `http://127.0.0.1:5000` in your web browser.

---

## 🧠 The ML Algorithm Under the Hood

The model runs **Gradient Descent** over a normalized multiple feature space.

1. **Min-Max Scaling**: Normalizes inputs to ensure numerical stability and speed up gradient descent:
   $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
2. **Inference Function**:
   $$\hat{y} = X \cdot w + b$$
3. **Loss Function (MSE)**:
   $$J(w, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2$$
4. **Parameter Gradients**:
   $$\frac{\partial J}{\partial w} = \frac{2}{n} X^T (\hat{y} - y)$$
   $$\frac{\partial J}{\partial b} = \frac{2}{n} \sum (\hat{y} - y)$$
5. **Update Rules**:
   $$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$
   $$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
