import torch
import torchvision
import torchvision.transforms as transforms
import os
from Network import Net, AddColorFeatures, LimitImageSize
from sklearn.metrics import recall_score

# Chemin du modèle pré-entraîné
MODEL_PATH = "trained_model3.pth"

# Dataset et transform
transform = transforms.Compose([
    LimitImageSize(max_size=3000),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    AddColorFeatures(),
    transforms.Normalize(mean=[0.5] * 6, std=[0.5] * 6)
])

dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corrected_wildfires_dataset')
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

# Split train/test comme dans l'entraînement
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Charger le modèle
net = Net(in_channels=6)
net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
net.eval()

all_targets = []
all_preds = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        all_targets.extend(targets.numpy())
        all_preds.extend(predicted.numpy())

# Calcul du recall
recall = recall_score(all_targets, all_preds, average='macro')
print(f"Recall du modèle sur le test set : {recall:.4f}")
