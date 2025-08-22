import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import os

def extract_hsv_palette(image_path, num_colors):
    """Extracts color palette in HSV and returns feature vector."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  
    
    img_hsv = img.reshape((-1, 3))  
    
    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(img_hsv)
    colors_hsv = kmeans.cluster_centers_

    colors_hsv = np.round(colors_hsv).astype(int)

    return colors_hsv

def process_folder(folder_path, output_csv_path, party_name, num_colors=5):
    data = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for idx, filename in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] processing: {filename}")

        image_path = os.path.join(folder_path, filename)

        try:
            colors_hsv = extract_hsv_palette(image_path, num_colors)
            hsv_vector = colors_hsv.flatten()

            entry = {
                'party': party_name,
                'filename': filename,
            }
            for i, value in enumerate(hsv_vector):
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
