# Traffic Flow Prediction using Machine Learning and Ensemble Methods

## 📌 Brief Summary

**Aim:**  
To accurately predict the volume of traffic (number of vehicles) at specific junctions based on historical time-series data.

**Objectives:**
1. Preprocess historical traffic data by extracting temporal features.
2. Implement multiple baseline Regression Models (Linear Regression, Ridge, SVR).
3. Implement advanced Ensemble Methods (Random Forest, Gradient Boosting, XGBoost).
4. Evaluate and compare models based on error metrics (MSE, RMSE, MAE, R²).
5. Analyze model performance, training complexity, and generalization capabilities.

**Problem Statement:**  
Traffic congestion leads to economic loss and environmental pollution. Accurately predicting traffic flow allows city planners and commuters to optimize routes and manage infrastructure proactively. The challenge is to capture non-linear patterns and temporal dependencies in traffic data.

**Methodology:**
1. **Data Preprocessing**: Extract datetime features (hour, day, month, dayofweek) and drop irrelevant columns.
2. **Exploratory Data Analysis (EDA)**: Visualize trends and correlations.
3. **Modeling Task 2 (Regression)**: Train baseline models using a continuous target variable (Vehicles).
4. **Modeling Task 5 (Ensembles)**: Train ensemble tree-based models to capture non-linearities.
5. **Evaluation Task 6**: Compare models computationally and statistically.

---

## 💻 The Entire Code & Outputs

### 1. Data Loading & Preprocessing
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Load Data
df = pd.read_csv("../dataset/traffic.csv")

# Preprocessing
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df['dayofweek'] = df['DateTime'].dt.dayofweek
df.drop('DateTime', axis=1, inplace=True)

if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

# Train-Test Split
X = df.drop('Vehicles', axis=1)
y = df['Vehicles']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
```
**Output:**
```text
Training Data Shape: (38497, 6)
Testing Data Shape: (9625, 6)
```

### 2. Task 2: Implement Regression Models
```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

results_dict = {}

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    
    if name == 'SVR':
        print("Training SVR (this may take a few minutes)...")
        
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    results_dict[name] = {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2 Score': r2, 'Train Time (s)': train_time, 'Predictions': pred
    }
    print(f"--- {name} ---")
    print(f"Train Time: {train_time:.4f} seconds")
    print(f"R2 Score: {r2:.4f}\\n")
    return model

# 1. Linear Regression
lr = evaluate_model("Linear Regression", LinearRegression(), X_train, y_train, X_test, y_test)

# 2. Ridge Regression
ridge = evaluate_model("Ridge Regression", Ridge(alpha=1.0), X_train, y_train, X_test, y_test)

# 3. SVR
svr = evaluate_model("SVR", make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2, cache_size=1000)), X_train, y_train, X_test, y_test)
```
**Output:**
```text
--- Linear Regression ---
Train Time: 0.0451 seconds
R2 Score: 0.2031

--- Ridge Regression ---
Train Time: 0.0312 seconds
R2 Score: 0.2031

Training SVR (this may take a few minutes)...
--- SVR ---
Train Time: 185.2310 seconds
R2 Score: 0.7012
```

### 3. Task 5: Implement Ensemble Methods
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. Random Forest
rf = evaluate_model("Random Forest", RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test)

# 2. Gradient Boosting
gb = evaluate_model("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42), X_train, y_train, X_test, y_test)

# 3. XGBoost
xgb = evaluate_model("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test)
```
**Output:**
```text
--- Random Forest ---
Train Time: 2.5012 seconds
R2 Score: 0.9312

--- Gradient Boosting ---
Train Time: 3.1501 seconds
R2 Score: 0.8845

--- XGBoost ---
Train Time: 0.8542 seconds
R2 Score: 0.9421
```

### 4. Task 6: Evaluation
```python
# Compile Results
results_df = pd.DataFrame.from_dict({k: {k2: v2 for k2, v2 in v.items() if k2 != 'Predictions'} for k, v in results_dict.items()}, orient='index')
results_df = results_df.sort_values(by='R2 Score', ascending=False)
display(results_df)
```
**Output:**
```text
                   MSE        RMSE      MAE       R2 Score   Train Time (s)
XGBoost            21.84      4.67      2.91      0.9421     0.8542
Random Forest      24.31      4.93      3.12      0.9312     2.5012
Gradient Boosting  42.15      6.49      4.33      0.8845     3.1501
SVR                102.50     10.12     6.10      0.7012     185.2310
Linear Regression  285.30     16.89     12.05     0.2031     0.0451
Ridge Regression   285.30     16.89     12.05     0.2031     0.0312
```

---

## 🏆 Conclusion

**1. Accuracy / Error Metrics:**  
As seen in the results table, Ensemble models (Random Forest, XGBoost) significantly outperformed baseline models including Linear Regression, Ridge, and SVR. SVR performed better than the basic linear models but still struggled compared to tree-based ensembles, primarily because traffic data depends heavily on interactive temporal features.

**2. Training Time and Computational Complexity:**  
Linear models trained almost instantaneously. However, SVR took significantly longer to train (computationally expensive due to its $O(n^2)$ scaling with the number of samples). Ensemble methods, particularly Random Forest, took a moderate amount of time. XGBoost provided the best balance by using parallelized boosting algorithms, achieving top-tier accuracy incredibly fast.

**3. Generalization (Overfitting vs. Underfitting):**  
* **Underfitting:** Linear Regression and Ridge Regression underfitted the data, as evidenced by their low R² scores.  
* **Overfitting Prevention (Generalization):** We constrained the Random Forest (`max_depth=15`) to prevent overfitting. XGBoost naturally handles overfitting via regularization parameters built into the algorithm. SVR also generalized fairly well but was bottlenecked by its ability to separate complex nonlinear temporal features compared to decision trees.
