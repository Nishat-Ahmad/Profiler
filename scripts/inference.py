import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import sys
import os

# Add the current directory to path so we can import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import FairFaceVGG16

# Define the category mappings (must match dataset.py)
RACE_LABELS = {
    0: 'White', 
    1: 'Black', 
    2: 'Latino_Hispanic',
    3: 'East Asian', 
    4: 'Southeast Asian',
    5: 'Indian', 
    6: 'Middle Eastern'
}

def get_inference_transform():
    """Returns the transformation used for validation/inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def load_model(weights_path, device):
    """Loads the model architecture and weights."""
    model = FairFaceVGG16(num_races=len(RACE_LABELS))
    
    # Load weights
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Weights file not found at {weights_path}")
        sys.exit(1)
        
    model.to(device)
    model.eval() # Set to evaluation mode
    return model

def predict(image_path, model, device, transform):
    """Runs inference on a single image."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Load and Preprocess Image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension -> [1, 3, 224, 224]
        input_tensor = input_tensor.to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 2. Run Model
    with torch.no_grad():
        gender_prob, age_pred, race_logits = model(input_tensor.to(device))

    # 3. Decode Outputs
    
    # Gender (Sigmoid output): < 0.5 is Male, >= 0.5 is Female
    # (Based on dataset.py: Male=0.0, Female=1.0)
    gender_val = gender_prob.item()
    gender = "Female" if gender_val >= 0.5 else "Male"
    
    # Age (Regression output)
    age = age_pred.item()
    
    # Race (Argmax of logits)
    race_idx = torch.argmax(race_logits, dim=1).item()
    race = RACE_LABELS.get(race_idx, "Unknown")
    
    # 4. Print Results
    print("-" * 30)
    print(f"Image:  {image_path}")
    print(f"Gender: {gender} ({gender_val:.2f})")
    print(f"Age:    {age:.1f} years")
    print(f"Race:   {race}")
    print("-" * 30)

if __name__ == "__main__":
    # Get the project root directory (one level up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_weights = os.path.join(project_root, "weights", "fairface_vgg16_weights.pth")

    parser = argparse.ArgumentParser(description="FairFace Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to the face image")
    parser.add_argument("--weights", type=str, default=default_weights, help="Path to model weights")
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = load_model(args.weights, device)
    
    # Predict
    transform = get_inference_transform()
    predict(args.image, model, device, transform)
