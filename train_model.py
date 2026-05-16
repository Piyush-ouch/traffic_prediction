import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Loading data...")
df = pd.read_csv("dataset/traffic.csv")

# Extract features
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df['dayofweek'] = df['DateTime'].dt.dayofweek

X = df[['Junction', 'year', 'month', 'day', 'hour', 'dayofweek']]
y = df['Vehicles']

print("Training RandomForest model...")
model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X, y)

print("Saving model to models/traffic_model.pkl...")
joblib.dump(model, "models/traffic_model.pkl")
print("Done!")
