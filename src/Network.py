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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## Convolutional layers, where weights represent conv kernels
        # 3 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        # 6 input channels (the output of the last layer), 16 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        ## Linear layer: MLP, i.e. fully-connected layer.
        self.fc1 = nn.Linear(in_features = 52*52, out_features = 120)  # 6*6 from the image dimension, and 16 for the number of channels
        self.fc2 = nn.Linear(in_features = 120, out_features = 84) # 120 is output of the previous layer.
        self.fc3 = nn.Linear(in_features = 84, out_features = 2) # 84 is the output of the previous layer, 10 is the number of classes.

    def forward(self, x):
        # Conv1, then max pooling over a (4, 4) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 4)
        # Conv2, then max pooling over a (4, 4) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 4) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x)) # Reshape each image, processed by conv, into a vector (required for linear layers)
        # 1st Linear layer
        x = F.relu(self.fc1(x))
        # 2nd Linear layer
        x = F.relu(self.fc2(x))
        # 3rd Linear layer
        x = self.fc3(x)
        return x

    def num_flat_features(self, x): # Computes the number of flat (*"vectorized"*) features from a 2D conv.
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    # Ce code ne s'exécute que si le fichier est lancé directement
    # (pas lors de l'import dans train.py)
    
    transform = transforms.Compose([
        LimitImageSize(max_size=3000),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5])
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

    net = Net()
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

    torch.save(net.state_dict(), "trained_model.pth")
