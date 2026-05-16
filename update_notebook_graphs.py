import json

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "# 1. Plot Actual vs Predicted for Linear Regression\n",
    "lr_preds = results_dict['Linear Regression']['Predictions']\n",
    "\n",
    "plt.figure(figsize=(15, 5))\n",
    "plt.plot(y_test.values[:150], label='Actual', marker='o', linestyle='-', alpha=0.7, color='blue')\n",
    "plt.plot(lr_preds[:150], label='Predicted (Linear Regression)', marker='x', linestyle='--', alpha=0.7, color='red')\n",
    "plt.title(\"Actual vs Predicted Traffic Volume (First 150 samples) - Linear Regression\")\n",
    "plt.xlabel(\"Sample Index\")\n",
    "plt.ylabel(\"Vehicles\")\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "# 2. Plot Actual vs Predicted for Random Forest\n",
    "rf_preds = results_dict['Random Forest']['Predictions']\n",
    "\n",
    "plt.figure(figsize=(15, 5))\n",
    "plt.plot(y_test.values[:150], label='Actual', marker='o', linestyle='-', alpha=0.7, color='blue')\n",
    "plt.plot(rf_preds[:150], label='Predicted (Random Forest)', marker='x', linestyle='--', alpha=0.7, color='green')\n",
    "plt.title(\"Actual vs Predicted Traffic Volume (First 150 samples) - Random Forest\")\n",
    "plt.xlabel(\"Sample Index\")\n",
    "plt.ylabel(\"Vehicles\")\n",
    "plt.legend()\n",
    "plt.show()\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Plot Actual vs Predicted values for the best model" in source or "Plot Actual vs Predicted for Linear Regression" in source:
            cell['source'] = new_source
            cell['outputs'] = [] # clear outputs to be safe
            break

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Modified notebook successfully.")
