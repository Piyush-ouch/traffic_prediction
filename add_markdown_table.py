import json

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Model Performance: Training vs Testing Accuracy\n",
        "\n",
        "| Model | Training Accuracy (R²) | Testing Accuracy (R²) | Overfitting Analysis |\n",
        "| :--- | :--- | :--- | :--- |\n",
        "| **Linear Regression** | 60.18% | 59.97% | **Underfitting** (Model is too simple) |\n",
        "| **Ridge Regression** | 60.18% | 59.97% | **Underfitting** (Same as LR) |\n",
        "| **SVR** | 86.81% | 86.73% | **Well Balanced** (But incredibly slow to train) |\n",
        "| **Gradient Boosting** | 90.25% | 90.17% | **Well Balanced** (Good generalization) |\n",
        "| **XGBoost** | 94.55% | 94.09% | **Excellent Balance** (High accuracy, no overfitting) |\n",
        "| **Random Forest** | **98.37%** | **96.41%** | **Slight Overfitting** (But still the highest overall performer) |\n"
    ]
}

# Find the index of the comparison code cell
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and "Training vs Testing Accuracy Comparison" in "".join(cell.get('source', [])):
        insert_idx = i
        break

if insert_idx is not None:
    # Check if already inserted
    already_inserted = False
    if insert_idx > 0 and nb['cells'][insert_idx-1]['cell_type'] == 'markdown':
        if "Model Performance: Training vs Testing Accuracy" in "".join(nb['cells'][insert_idx-1]['source']):
            already_inserted = True
    
    if not already_inserted:
        nb['cells'].insert(insert_idx, markdown_cell)
        with open('notebooks/updated_traffic_prediction_v2.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Markdown table inserted successfully.")
    else:
        print("Markdown table already exists.")
else:
    print("Could not find the target cell.")
