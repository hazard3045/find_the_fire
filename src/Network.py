import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

class LimitImageSize:
    def __init__(self, max_size=3000):
        self.max_size = max_size
    def __call__(self, img):
        if img.width > self.max_size or img.height > self.max_size:
            img = img.resize((224, 224), Image.BILINEAR)
        return img


class BoostRed:
    def __init__(self, factor=2.0):
        self.factor = factor

    def __call__(self, img_tensor):
        img_tensor = img_tensor.clone()
        img_tensor[0] = img_tensor[0] * self.factor  # canal R amplifié
        return img_tensor


class AddColorFeatures:
    def __init__(self, eps=1e-6):
        self.eps = eps
    def __call__(self, t):
        # t: Tensor [3, H, W], values in [0,1]
        R, G, B = t[0], t[1], t[2]
        sumc = R + G + B + self.eps
        norm_red = R / sumc                     # R / (R+G+B)
        excess_red = torch.clamp(2*R - G - B, 0.0, 1.0)  # index simple
        rg_diff = torch.clamp((R - G + 1.0) / 2.0, 0.0, 1.0)  # red vs green
        extra = torch.stack([norm_red, excess_red, rg_diff], dim=0)
        return torch.cat([t, extra], dim=0)  # shape [6,H,W]


class Net(nn.Module):
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Global pooling -> fixe la taille de sortie quel que soit l'input
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)           # shape (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # shape (batch, 128)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Ce code ne s'exécute que si le fichier est lancé directement
    # (pas lors de l'import dans train.py)
    
    transform = transforms.Compose([
        LimitImageSize(max_size=3000),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
 #       BoostRed(factor=1.0),
 #       AddColorFeatures(),
        # Normaliser 6 canaux (3 RGB + 3 extras). Moyennes/std simples à 0.5
        transforms.Normalize(mean=[0.5] * 3,
                             std=[0.5] * 3)
    ])

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corrected_wildfires_dataset')
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # On crée le modèle en tenant compte des 6 canaux produits par AddColorFeatures
    net = Net(in_channels=3)
    out = net(images)

    # Entraînement du réseau
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 5

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}] Loss moyen: {epoch_loss:.4f}")

    # Évaluation sur le test set
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    test_acc = 100 * correct / total
    print(f"Test accuracy: {test_acc:.2f}%")

    # Sauvegarde du modèle entraîné
    torch.save(net.state_dict(), "trained_model2.pth")