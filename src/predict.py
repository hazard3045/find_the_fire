import torch
import torchvision.transforms as transforms
from PIL import Image
from Network import Net
import sys

# Configuration
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations (mêmes que pour l'entraînement)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Charger le modèle
net = Net()
net.fc3 = torch.nn.Linear(in_features=84, out_features=2)  # 2 classes
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net = net.to(DEVICE)
net.eval()

classes = ['fire', 'nofire']

def predict_image(image_path):
    """Prédire la classe d'une image"""
    try:
        # Charger et transformer l'image
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)  # Ajouter dimension batch
        image_tensor = image_tensor.to(DEVICE)
        
        # Prédiction
        with torch.no_grad():
            outputs = net(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item() * 100
        
        print(f"\n📷 Image: {image_path}")
        print(f"🔥 Prédiction: {predicted_class.upper()}")
        print(f"📊 Confiance: {confidence_score:.2f}%")
        
        # Afficher les probabilités pour chaque classe
        print("\nProbabilités détaillées:")
        for i, class_name in enumerate(classes):
            prob = probabilities[0][i].item() * 100
            print(f"  - {class_name}: {prob:.2f}%")
        
        return predicted_class, confidence_score
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {str(e)}")
        return None, 0.0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <chemin_image>")
        print("Exemple: python predict.py ../corrected_wildfires_dataset/fire/image1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_image(image_path)
