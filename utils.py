# utils.py
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class PlantDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]["image"])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx]["label"])
        return self.transform(image), label
