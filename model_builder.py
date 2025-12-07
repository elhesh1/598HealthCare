import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.functional as TF   
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




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

        # transorming like the paper 
        img = TF.resize(img, [224, 224])
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        label = torch.tensor(self.y[idx], dtype=torch.float32)

        return img, label





# for evaluating model - called after each epoch and then again on the testing data
def evaluate_model(model, loader):
    model.eval()

    preds, labels = [], []

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

# for training each epoch
def one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to("cuda" if torch.cuda.is_available() else "cpu")
        labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")

        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def train_melanoma_model(X, y, save_path="melanoma_model.pth"):

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    min_length = min(len(pos_idx), len(neg_idx))
    shortest = np.concatenate([pos_idx[:min_length], neg_idx[:min_length]])

    np.random.shuffle(shortest) # 

    X = X[shortest]
    y = y[shortest]

    split = int(len(y) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(CurrDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(CurrDataset(X_test, y_test), batch_size=32, shuffle=False)

    #Resnet18 instead of 50 bc my computer is weak
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # The "meat" of the model
    for epoch in range(5):

        loss = one_epoch(model, train_loader, criterion, optimizer)
        cm, acc = evaluate_model(model, train_loader)

        print("Epoch " + str(epoch + 1) + "/5")
        print("Loss:", loss)
        print("Train cm:" )
        print(cm)
        print("Accuracy:", acc)


    cm, acc = evaluate_model(model, test_loader)
    print("Test cm:")
    print(cm)
    print("Test Accuracy:", acc)

    torch.save(model.state_dict(), save_path)

    return model, (cm, acc)
