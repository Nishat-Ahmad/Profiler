import sys
import os
import torch

# Add scripts folder to path so we can import the model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'scripts'))

try:
    from model import FairFaceVGG16
except ImportError:
    # Fallback if running from a different directory context
    sys.path.append('scripts')
    from model import FairFaceVGG16

try:
    from torchsummary import summary
except ImportError:
    print("\n" + "!"*60)
    print("MISSING LIBRARY: torchsummary")
    print("To fix this, run: pip install torchsummary")
    print("!"*60 + "\n")
    sys.exit(1)

def show_comparison():
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = FairFaceVGG16()
    model = model.to(device)
    
    print("\n" + "="*30 + " AUTOMATED SUMMARY " + "="*30)
    # VGG16 standard input size is (3, 224, 224)
    summary(model, (3, 224, 224))

if __name__ == "__main__":
    show_comparison()
