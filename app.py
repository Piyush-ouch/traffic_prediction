from flask import Flask, render_template, request
import joblib
import numpy as np
import datetime
import os

app = Flask(__name__)

# Load model
if os.path.exists("models/traffic_model.pkl"):
    model = joblib.load("models/traffic_model.pkl")
else:
    model = None
    print("WARNING: models/traffic_model.pkl not found. Please run train_model.py first.")

# In-memory store for last 5 predictions
prediction_history = []

@app.route('/')
def home():
    return render_template('index.html', history=prediction_history)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', result={"error": "Model not trained yet.", "success": False}, history=prediction_history)

    try:
        junction = int(request.form['junction'])
        date_str = request.form['date']
        time_str = request.form['time']

        dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        dayofweek = dt.weekday()

        features = np.array([[junction, year, month, day, hour, dayofweek]])
        prediction = model.predict(features)[0]

        # Determine traffic level
        if prediction < 20:
            level = "LOW"
            color = "#10b981"
        elif prediction < 50:
            level = "MODERATE"
            color = "#f59e0b"
        elif prediction < 100:
            level = "HIGH"
            color = "#ef4444"
        else:
            level = "SEVERE"
            color = "#7f1d1d"

        result = {
            "prediction": f"{prediction:.0f}",
            "level": level,
            "color": color,
            "success": True
        }

        # Save to history (keep last 5)
        label = f"J{junction} {dt.strftime('%d/%m %H:%M')}"
        prediction_history.append({"label": label, "value": round(float(prediction), 1), "color": color})
        if len(prediction_history) > 5:
            prediction_history.pop(0)

    except Exception as e:
        result = {"error": str(e), "success": False}

    return render_template('index.html', result=result, history=prediction_history)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
