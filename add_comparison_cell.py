import json

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training vs Testing Accuracy Comparison\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "models = list(results_df.index)\n",
        "train_r2 = results_df['Training R2'].values\n",
        "test_r2 = results_df['Testing R2'].values\n",
        "\n",
        "x = np.arange(len(models))\n",
        "width = 0.35\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "rects1 = ax.bar(x - width/2, train_r2, width, label='Training Accuracy (R2)', color='teal')\n",
        "rects2 = ax.bar(x + width/2, test_r2, width, label='Testing Accuracy (R2)', color='coral')\n",
        "\n",
        "ax.set_ylabel('Accuracy (R2 Score)')\n",
        "ax.set_title('Overfitting Analysis: Training vs Testing Accuracy by Model')\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(models, rotation=45)\n",
        "ax.legend(loc='lower left')\n",
        "ax.set_ylim(0, 1.1)  # Give some headroom for the legend\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('../graphs/training_vs_testing_comparison.png', bbox_inches='tight')\n",
        "plt.show()\n"
    ]
}

# We want to insert this cell before the "Plot Actual vs Predicted" cell
insert_idx = len(nb['cells']) - 1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and "Plot Actual vs Predicted" in "".join(cell['source']):
        insert_idx = i
        break

# Check if we already inserted it
already_inserted = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "Training vs Testing Accuracy Comparison" in "".join(cell['source']):
        already_inserted = True
        cell['source'] = new_cell['source']  # Update if exists
        break

if not already_inserted:
    nb['cells'].insert(insert_idx, new_cell)

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Inserted comparison cell successfully.")
