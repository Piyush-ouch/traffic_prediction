import json

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "# Plot Actual vs Predicted for Multiple Models\n",
    "def plot_actual_vs_predicted(model_name, color, filename):\n",
    "    if model_name not in results_dict:\n",
    "        print(f\"Model {model_name} not found in results_dict.\")\n",
    "        return\n",
    "    preds = results_dict[model_name]['Predictions']\n",
    "    plt.figure(figsize=(15, 5))\n",
    "    plt.plot(y_test.values[:150], label='Actual', marker='o', linestyle='-', alpha=0.7, color='blue')\n",
    "    plt.plot(preds[:150], label=f'Predicted ({model_name})', marker='x', linestyle='--', alpha=0.7, color=color)\n",
    "    plt.title(f\"Actual vs Predicted Traffic Volume (First 150 samples) - {model_name}\")\n",
    "    plt.xlabel(\"Sample Index\")\n",
    "    plt.ylabel(\"Vehicles\")\n",
    "    plt.legend()\n",
    "    plt.savefig(f\"../graphs/{filename}\", bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "plot_actual_vs_predicted('Linear Regression', 'red', 'actual_vs_predicted_lr.png')\n",
    "plot_actual_vs_predicted('Ridge Regression', 'orange', 'actual_vs_predicted_ridge.png')\n",
    "plot_actual_vs_predicted('SVR', 'purple', 'actual_vs_predicted_svr.png')\n",
    "plot_actual_vs_predicted('Random Forest', 'green', 'actual_vs_predicted_rf.png')\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Plot Actual vs Predicted for Multiple Models" in source:
            cell['source'] = new_source
            cell['outputs'] = [] # clear outputs to be safe
            break

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Modified notebook successfully.")
