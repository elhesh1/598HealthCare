import numpy as np
from model_builder import train_melanoma_model

X = np.load("X_lesion_high.npy", mmap_mode="r")
y = np.load("y_lesion.npy")

model, results = train_melanoma_model( X, y, 
    save_path="melanoma_resnet18_lesion_high.pth"
)
