from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import os
import re

def extract_rgb_palette(image_path, num_colors):
    """Extracts color palette and returns feature vector."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img.width // 2, img.height // 2)) 
    img_array = np.array(img)

    img_rgb = img_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(img_rgb)
    colors_rgb = kmeans.cluster_centers_
    colors_rgb = np.round(colors_rgb).astype(int)

    return colors_rgb

def process_folder(folder_path, output_csv_path, party_name, num_colors):
    data = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for idx, filename in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] process: {filename}")

        image_path = os.path.join(folder_path, filename)

        try:
            colors_rgb = extract_rgb_palette(image_path, num_colors)
            rgb_vector = colors_rgb.flatten()

            entry = {
                'party': party_name,
                'filename': filename,
            }
            for i, value in enumerate(rgb_vector):
                entry[f'feature_{i}'] = value
            data.append(entry)

        except Exception as e:
            print(f"error: {filename}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV saved: {output_csv_path}")

party_name = "spd"
folder_path = r'C:\Users\lukaspasold\Database\spdde\jpg'
output_csv_path = r'C:\Users\lukaspasold\colors\colors_spd.csv'

process_folder(folder_path, output_csv_path, party_name, num_colors=4)
