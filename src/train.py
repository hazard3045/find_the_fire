import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, has_file_allowed_extension, IMG_EXTENSIONS
from torch.utils.data import Dataset
import os
from Network import Net

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
# Résoudre le chemin du dataset par rapport à ce fichier afin d'éviter
# les problèmes liés au répertoire de travail courant.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'corrected_wildfires_dataset')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations pour les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner pour correspondre au réseau
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation pour 3 canaux
])


class FilteredImageFolder(Dataset):
    """ImageFolder qui ignore les sous-dossiers sans images valides.
    Évite l'erreur FileNotFoundError quand il y a des dossiers utilitaires
    (comme 'changes') dans le répertoire du dataset.
    """
    def __init__(self, root, transform=None, loader=default_loader, extensions=IMG_EXTENSIONS):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.extensions = extensions

        # Découvrir les dossiers de classes contenant au moins une image valide
        classes = []
        class_to_idx = {}
        samples = []

        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if not os.path.isdir(path):
                continue

            # Collecter les fichiers images dans ce dossier
            files = [f for f in sorted(os.listdir(path)) 
                    if has_file_allowed_extension(f, self.extensions)]
            if len(files) == 0:
                # Ignorer les dossiers vides/sans images
                print(f"  ⚠️  Dossier ignoré (pas d'images): {entry}")
                continue

            classes.append(entry)
            class_idx = len(class_to_idx)
            class_to_idx[entry] = class_idx

            for fname in files:
                item = (os.path.join(path, fname), class_idx)
                samples.append(item)

        if len(classes) == 0:
            raise RuntimeError(f"Aucune classe d'images valide trouvée dans '{root}'")

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# Charger le dataset
dataset = FilteredImageFolder(root=DATA_DIR, transform=transform)
print(f"Nombre total d'images: {len(dataset)}")
print(f"Classes détectées: {dataset.classes}")

# Diviser en ensembles d'entraînement et de validation (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Images d'entraînement: {len(train_dataset)}")
print(f"Images de validation: {len(val_dataset)}")

# Initialiser le réseau
net = Net()
# Modifier la dernière couche pour 2 classes (fire/nofire) au lieu de 10
net.fc3 = nn.Linear(in_features=84, out_features=2)
net = net.to(DEVICE)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Fonction d'entraînement
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 10 == 9:  # Afficher toutes les 10 mini-batches
            print(f'  Batch [{i+1}/{len(loader)}], Loss: {running_loss/10:.3f}')
            running_loss = 0.0
    
    accuracy = 100 * correct / total
    return accuracy

# Fonction de validation
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Boucle d'entraînement
print(f"\nDébut de l'entraînement sur {DEVICE}")
print("-" * 50)

best_val_acc = 0.0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Entraînement
    train_acc = train_epoch(net, train_loader, criterion, optimizer, DEVICE)
    
    # Validation
    val_loss, val_acc = validate(net, val_loader, criterion, DEVICE)
    
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.2f}%")
    
    # Sauvegarder le meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), 'best_model.pth')
        print(f"✓ Meilleur modèle sauvegardé (accuracy: {val_acc:.2f}%)")

print("\n" + "="*50)
print(f"Entraînement terminé!")
print(f"Meilleure précision de validation: {best_val_acc:.2f}%")
print("="*50)

# Sauvegarder le modèle final
torch.save(net.state_dict(), 'final_model.pth')
print("Modèle final sauvegardé dans 'final_model.pth'")
