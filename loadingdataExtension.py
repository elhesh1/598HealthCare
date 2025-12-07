import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

csv_path = r"ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"
image_dir = r"ISIC2018_Task3_Training_Input\ISIC2018_Task3_Training_Input"

df_labels = pd.read_csv(csv_path)
df_labels['image_id'] = df_labels['image'].str.replace('.jpg', '', regex=False)
df_labels['image_id'] = df_labels['image_id'].str.replace('.png', '', regex=False)

image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
              glob.glob(os.path.join(image_dir, "*.png"))

df_imgs = pd.DataFrame({'full_path': image_paths})
df_imgs["filename"] = df_imgs["full_path"].str.split("\\").str[-1]
df_imgs["image_id"] = df_imgs["filename"].str.replace(r"\..*$", "", regex=True)
df_imgs = df_imgs.drop(columns=["filename"])

df = df_imgs.merge(df_labels, on='image_id', how='left')

# the three malignant ones
df["malignant"] = ( (df["MEL"] == 1.0) | (df["BCC"] == 1.0) | (df["AKIEC"] == 1.0)).astype(int)

# 
df.to_csv("isic_task3_binary.csv", index=False)

X_list = []
Y_list = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        img = Image.open(row["full_path"]).convert("RGB")
        # resize to 224 for consistency and so it fits in memory
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.uint8)
    except Exception as e:
        continue

    X_list.append(img)
    Y_list.append(int(row["malignant"]))

X_whole = np.array(X_list, dtype=np.uint8)
Y_whole = np.array(Y_list, dtype=np.int64)

np.save("X_whole_task3.npy", X_whole)
np.save("Y_whole_task3.npy", Y_whole)
