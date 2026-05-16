from flask import Flask, render_template, request
import joblib
import numpy as np
import datetime
import os

app = Flask(__name__)

# Load model if exists, otherwise it will crash and user must run train_model.py first
if os.path.exists("models/traffic_model.pkl"):
    model = joblib.load("models/traffic_model.pkl")
else:
    model = None
    print("WARNING: models/traffic_model.pkl not found. Please run train_model.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', result={"error": "Model not trained yet.", "success": False})

    try:
        junction = int(request.form['junction'])
        date_str = request.form['date'] # format: YYYY-MM-DD
        time_str = request.form['time'] # format: HH:MM
        
        # Parse datetime
        dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        dayofweek = dt.weekday() # Monday is 0, Sunday is 6
        
        features = np.array([[junction, year, month, day, hour, dayofweek]])
        prediction = model.predict(features)[0]
        
        # Determine traffic level
        if prediction < 20:
            level = "LOW"
            color = "#10b981" # Emerald green
        elif prediction < 50:
            level = "MODERATE"
            color = "#f59e0b" # Amber
        elif prediction < 100:
            level = "HIGH"
            color = "#ef4444" # Red
        else:
            level = "SEVERE"
            color = "#7f1d1d" # Dark red
            
        result = {
            "prediction": f"{prediction:.0f}",
            "level": level,
            "color": color,
            "success": True
        }
        
    except Exception as e:
        result = {
            "error": str(e),
            "success": False
        }

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
