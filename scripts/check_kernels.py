import sys
import os
import torch
from torchvision.models import vgg16, VGG16_Weights

# Add the scripts directory relative to this file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'scripts'))
from model import FairFaceVGG16

def count_kernels():
    model = FairFaceVGG16()
    print(f"\n{'-'*65}")
    print(f"{'Layer':<20} | {'Kernels (Out Channels)':<25} | {'Kernel Size'}")
    print(f"{'-'*65}")
    
    total_kernels = 0
    conv_layer_count = 0

    # Only count kernels in the 'features' part (VGG16 backbone)
    # The classifiers are fully connected layers, not convolutional kernels.
    for name, module in model.features.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layer_count += 1
            print(f"{name:<20} | {module.out_channels:<25} | {module.kernel_size}")
            total_kernels += module.out_channels
            
    print(f"{'-'*65}")
    print(f"Total Convolutional Layers: {conv_layer_count}")
    print(f"Total Kernels (Filters) in VGG16 Backbone: {total_kernels}")
    print(f"{'-'*65}")

if __name__ == "__main__":
    count_kernels()
