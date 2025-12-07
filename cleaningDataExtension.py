import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CSV_PATH = "isic_task3_binary.csv"

df = pd.read_csv(CSV_PATH)

X_list = []
Y_list = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = row["full_path"]
    try:
        img = np.array(Image.open(path).convert("RGB"))
    except:
        continue

    X_list.append(img.astype(np.uint8))
    Y_list.append(int(row["malignant"]))

X_whole = np.array(X_list, dtype=np.uint8)
Y_whole = np.array(Y_list, dtype=np.int64)

np.save("X_whole_task3.npy", X_whole)
np.save("Y_whole_task3.npy", Y_whole)

