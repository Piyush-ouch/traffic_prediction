import json
import base64
import os

os.makedirs('graphs', exist_ok=True)

with open('notebooks/updated_traffic_prediction_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

img_count = 1
image_names = ['feature_importance.png', 'metrics_comparison.png', 'actual_vs_predicted_lr.png', 'actual_vs_predicted_rf.png']

for cell in nb.get('cells', []):
    for output in cell.get('outputs', []):
        if 'data' in output and 'image/png' in output['data']:
            img_data = output['data']['image/png']
            name = image_names[img_count-1] if img_count <= len(image_names) else f'graph_{img_count}.png'
            with open(f'graphs/{name}', 'wb') as f_img:
                f_img.write(base64.b64decode(img_data))
            print(f"Saved graphs/{name}")
            img_count += 1
