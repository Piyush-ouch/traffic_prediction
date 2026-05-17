import json

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def evaluate_model(" in source and "train_pred =" not in source:
            new_source = source.replace(
                "    pred = model.predict(X_test)\n",
                "    train_pred = model.predict(X_train)\n"
                "    train_r2 = r2_score(y_train, train_pred)\n"
                "    pred = model.predict(X_test)\n"
            ).replace(
                "    print(f\"R2 Score: {r2:.4f}\\n\")\n",
                "    print(f\"Training R2 Score: {train_r2:.4f}\")\n"
                "    print(f\"Testing R2 Score: {r2:.4f}\\n\")\n"
            ).replace(
                "'R2 Score': r2,",
                "'Training R2': train_r2, 'Testing R2': r2,"
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            # Fix the last extra newline if it was added incorrectly
            if cell['source'][-1] == '\n':
                cell['source'] = cell['source'][:-1]
                
        # Update the compilation cell if it exists
        if "results_df.sort_values(by='R2 Score'" in source:
            new_source = source.replace(
                "results_df.sort_values(by='R2 Score'",
                "results_df.sort_values(by='Testing R2'"
            ).replace(
                "y='R2 Score'",
                "y='Testing R2'"
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source'][-1] == '\n':
                cell['source'] = cell['source'][:-1]

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
