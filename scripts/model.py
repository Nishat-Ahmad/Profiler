import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class FairFaceVGG16(nn.Module):
    def __init__(self, num_races=7):
        super(FairFaceVGG16, self).__init__()
        
        # 1. Load pretrained VGG16 backbone
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Extract features
        self.features = vgg.features
        
        # 2. Freeze pretrained layers
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 3. Modify avgpool (reduce spatial dimensions)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Flattened size: 512 channels * 2 * 2
        in_features = 512 * 2 * 2
        
        # Shared classifier base
        self.shared_classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        
        # 4. Create separate branches for each task
        
        # Gender classification (Binary) -> requires Sigmoid for BCELoss
        self.gender_classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Age prediction (Regression) -> L1Loss (MAE) requires raw continuous output
        self.age_regressor = nn.Sequential(
            nn.Linear(512, 1)
        )
        
        # Race classification (Multi-class)
        self.race_classifier = nn.Sequential(
            nn.Linear(512, num_races)
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Shared representations
        x = self.shared_classifier(x)
        
        # Branch outputs
        gender_out = self.gender_classifier(x)
        age_out = self.age_regressor(x)
        race_out = self.race_classifier(x)
        
        return gender_out, age_out, race_out

# Example instantiation and loss function definition
if __name__ == "__main__":
    model = FairFaceVGG16(num_races=7)
    
    # Losses
    criterion_gender = nn.BCELoss()
    criterion_age = nn.L1Loss()
    criterion_race = nn.CrossEntropyLoss()
    
    # Dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(8, 3, 224, 224)
    gender_pred, age_pred, race_pred = model(dummy_input)
    
    print("Model Output Shapes:")
    print(f"Gender: {gender_pred.shape}")
    print(f"Age:    {age_pred.shape}")
    print(f"Race:   {race_pred.shape}")
    
    # Dummy targets
    dummy_gender_target = torch.empty(8, 1).random_(2) # 0 or 1
    dummy_age_target = torch.randint(0, 80, (8, 1)).float()
    dummy_race_target = torch.empty(8, dtype=torch.long).random_(7)
    
    # Calculate dummy losses
    loss_gender = criterion_gender(gender_pred, dummy_gender_target)
    loss_age = criterion_age(age_pred, dummy_age_target)
    loss_race = criterion_race(race_pred, dummy_race_target)
    
    total_loss = loss_gender + loss_age + loss_race
    print(f"\nDummy Losses:")
    print(f"Gender Loss (BCE): {loss_gender.item():.4f}")
    print(f"Age Loss (L1):     {loss_age.item():.4f}")
    print(f"Race Loss (CE):    {loss_race.item():.4f}")
    print(f"Total Loss:        {total_loss.item():.4f}")