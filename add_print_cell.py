import json

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Dynamically print the Training vs Testing Accuracy in a text-based table format\n",
        "print(f\"{'-'*75}\")\n",
        "print(f\"{'Model':<20} | {'Training Accuracy (R2)':<25} | {'Testing Accuracy (R2)':<25}\")\n",
        "print(f\"{'-'*75}\")\n",
        "for model, row in results_df.iterrows():\n",
        "    train_r2 = f\"{row['Training R2'] * 100:.2f}%\"\n",
        "    test_r2 = f\"{row['Testing R2'] * 100:.2f}%\"\n",
        "    print(f\"{model:<20} | {train_r2:<25} | {test_r2:<25}\")\n",
        "print(f\"{'-'*75}\")\n"
    ]
}

# Find the index of the markdown table cell
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and "Model Performance: Training vs Testing Accuracy" in "".join(cell.get('source', [])):
        insert_idx = i
        break

if insert_idx is not None:
    # Check if we already inserted it
    already_inserted = False
    if insert_idx > 0 and nb['cells'][insert_idx-1]['cell_type'] == 'code':
        if "Dynamically print the Training vs Testing Accuracy" in "".join(nb['cells'][insert_idx-1]['source']):
            already_inserted = True
            nb['cells'][insert_idx-1]['source'] = new_cell['source'] # Update if it exists
    
    if not already_inserted:
        nb['cells'].insert(insert_idx, new_cell)
        
    with open('notebooks/updated_traffic_prediction_v2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Inserted dynamic print cell successfully.")
else:
    print("Could not find the target markdown cell.")
