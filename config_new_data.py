import numpy as np
from test_new_data import test_melanoma_model

X = np.load("X_whole_task3.npy")
y = np.load("Y_whole_task3.npy")

test_melanoma_model(X, y, "melanoma_resnet18_background.pth")
