import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from Network import Net

# Configuration
MODEL_PATH = 'best_model.pth'  # or 'final_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dénormaliser les images pour la visualisation
def preprocess_image(img_path):
    """Load and preprocess image for both model input and visualization."""
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((224, 224))
    
    # For visualization 
    img_array = np.array(img_resized) / 255.0
    
    # For model input 
    img_tensor = transform(img_resized).unsqueeze(0)
    
    return img_tensor, img_array


def load_model(model_path):
    model = Net()
    model.fc3 = torch.nn.Linear(in_features=84, out_features=2)
    
    # Load trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠️  Model file not found: {model_path}")
        sys.exit(1)
    
    model.to(DEVICE)
    model.eval()
    return model


def generate_gradcam(model, img_tensor, img_array, target_class=None, save_path=None):
    """
    Generate Grad-CAM heatmap for the given image.
    
    Args:
        model: The trained neural network
        img_tensor: Preprocessed image tensor for model input
        img_array: Original image as numpy array (0-1 range) for visualization
        target_class: Class to visualize (0=fire, 1=nofire). If None, uses predicted class.
        save_path: Path to save the visualization. If None, displays instead.
    """
    # la dernière couche de conv !!
    target_layers = [model.conv2]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor.to(DEVICE))
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Use predicted class if target_class not specified
    if target_class is None:
        target_class = predicted_class
    
    # Generate CAM
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=img_tensor.to(DEVICE), targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # supperpose le cam sur l'image
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    
    # Create figure with original image, heatmap, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap only with colorbar
    im = axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Rouge = Important, Bleu = Peu important)', 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    # Add colorbar to show scale
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Importance', rotation=270, labelpad=15)
    
    # Overlay
    axes[2].imshow(visualization)
    class_names = ['Fire', 'No Fire']
    axes[2].set_title(f'Overlay\nPredicted: {class_names[predicted_class]} ({confidence:.2%})', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()
    
    return predicted_class, confidence, grayscale_cam


def batch_process_directory(model, input_dir, output_dir):
    """
    Process all images in a directory and save Grad-CAM visualizations.
    
    Args:
        model: The trained neural network
        input_dir: Directory containing images to process
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if os.path.splitext(f.lower())[1] in valid_extensions]
    
    print(f"\nProcessing {len(image_files)} images from {input_dir}...")
    
    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(input_dir, filename)
        output_filename = f"gradcam_{os.path.splitext(filename)[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            img_tensor, img_array = preprocess_image(img_path)
            predicted_class, confidence, _ = generate_gradcam(
                model, img_tensor, img_array, save_path=output_path
            )
            class_names = ['Fire', 'No Fire']
            print(f"  [{i}/{len(image_files)}] {filename}: {class_names[predicted_class]} ({confidence:.2%})")
        except Exception as e:
            print(f"  ⚠️  Error processing {filename}: {e}")
    
    print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    """Main function to demonstrate Grad-CAM visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmaps for fire detection')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--dir', type=str, help='Directory containing images to process')
    parser.add_argument('--output', type=str, default='gradcam_output', 
                       help='Output directory for batch processing')
    parser.add_argument('--model', type=str, default=MODEL_PATH, 
                       help='Path to trained model weights')
    parser.add_argument('--target-class', type=int, choices=[0, 1], 
                       help='Target class for CAM (0=fire, 1=nofire). Default: predicted class')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    if args.image:
        # Process single image
        print(f"\nProcessing image: {args.image}")
        img_tensor, img_array = preprocess_image(args.image)
        
        # Generate and display
        output_path = f"gradcam_{os.path.basename(args.image)}"
        predicted_class, confidence, _ = generate_gradcam(
            model, img_tensor, img_array, 
            target_class=args.target_class,
            save_path=output_path
        )
        class_names = ['Fire', 'No Fire']
        print(f"Prediction: {class_names[predicted_class]} ({confidence:.2%})")
        
    elif args.dir:
        # Process directory
        batch_process_directory(model, args.dir, args.output)
        
    else:
        # Demo mode: process sample FIRE images only from dataset
        print("\nDemo mode: Processing FIRE images only from dataset...")
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dataset_dir = os.path.join(base_dir, 'corrected_wildfires_dataset')
        
        # Process only fire samples (not nofire)
        class_name = 'fire'
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_dir):
            output_dir = os.path.join('gradcam_output', class_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get first 5 images
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
            
            print(f"\nProcessing {len(images)} {class_name} samples...")
            for img_file in images:
                img_path = os.path.join(class_dir, img_file)
                output_path = os.path.join(output_dir, f"gradcam_{img_file}")
                
                try:
                    img_tensor, img_array = preprocess_image(img_path)
                    predicted_class, confidence, _ = generate_gradcam(
                        model, img_tensor, img_array, save_path=output_path
                    )
                    class_names = ['Fire', 'No Fire']
                    print(f"  {img_file}: {class_names[predicted_class]} ({confidence:.2%})")
                except Exception as e:
                    print(f"  ⚠️  Error: {e}")
        
        print("\n✓ Demo complete! Check gradcam_output/fire/ directory for results.")


if __name__ == "__main__":
    main()
