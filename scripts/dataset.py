import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Mapping for the 7 race categories
        self.race_mapping = {
            'White': 0, 'Black': 1, 'Latino_Hispanic': 2,
            'East Asian': 3, 'Southeast Asian': 4,
            'Indian': 5, 'Middle Eastern': 6
        }

    def __len__(self):
        return len(self.data_frame)

    def _parse_age(self, age_str):
        """Convert the age range string into a numerical regression target (midpoint)."""
        age_str = str(age_str).strip()
        if age_str == 'more than 70':
            return 75.0
        
        parts = age_str.split('-')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2.0
        
        try:
            return float(age_str)
        except ValueError:
            return 30.0 # Fallback default

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        
        # Construct the image path
        # Note: The 'file' column in the CSV usually looks like 'train/1.jpg' 
        # so root_dir should point to the FairFace folder
        img_path = os.path.join(self.root_dir, row['file'])
        
        # Open image and ensure it's RGB
        image = Image.open(img_path).convert('RGB')

        # 1. Gender: Male -> 0.0, Female -> 1.0 (Binary)
        gender = 0.0 if row['gender'] == 'Male' else 1.0
        gender_tensor = torch.tensor([gender], dtype=torch.float32)

        # 2. Age: Range midpoint (Regression)
        age = self._parse_age(row['age'])
        age_tensor = torch.tensor([age], dtype=torch.float32)

        # 3. Race: Categorical integer (Multi-class)
        race = self.race_mapping[row['race']]
        race_tensor = torch.tensor(race, dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, gender_tensor, age_tensor, race_tensor

# Common ImageNet transform for VGG16
def get_transform(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

if __name__ == "__main__":
    # Test dataset instantiation
    data_dir = "../FairFace"
    csv_path = os.path.join(data_dir, "fairface_label_train.csv")
    
    if os.path.exists(csv_path):
        dataset = FairFaceDataset(csv_file=csv_path, root_dir=data_dir, transform=get_transform(is_train=False))
        img, gen, age, race = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Gender: {gen.item()} (0=Male, 1=Female)")
        print(f"Age: {age.item()} (Midpoint)")
        print(f"Race Class: {race.item()}")
    else:
        print(f"Could not find {csv_path} - verify dataset path.")