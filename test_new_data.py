import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Same thing as model builder
class CurrDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = TF.resize(img, [224, 224])
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, label

# bascially using the same thing as model_builder
def evaluate_model(model, loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to("cuda" if torch.cuda.is_available() else "cpu")
            logits = torch.sigmoid(model(imgs).squeeze())
            pred = (logits > 0.5).float().cpu().numpy()

            preds.extend(pred)
            labels.extend(labs.numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    return cm, acc

def test_melanoma_model(X, y, model_path):
    test_loader = DataLoader(CurrDataset(X, y), batch_size=32, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    cm, acc = evaluate_model(model, test_loader)
    print("Confusion matrix:")
    print(cm)
    print("Accuracy:", acc)

    return cm, acc
