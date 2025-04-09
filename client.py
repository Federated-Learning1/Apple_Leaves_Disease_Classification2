# client.py
# import the libraries
import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import flwr as fl
from timm import create_model

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
NUM_CLASSES = len(CLASSES)

# Custom Dataset
class PlantDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.label_map = {label: idx for idx, label in enumerate(CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image"]
        label = self.label_map[self.data.iloc[idx]["label"]]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

# Training function
def train(model, trainloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    train_loss = total_loss / total
    train_acc = correct / total
    print(f"[Train] Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

# Federated client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, client_id):
        self.model = model.to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.client_id = client_id

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        train(self.model, self.trainloader, self.criterion, optimizer)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_total = 0, 0, 0.0

        with torch.no_grad():
            for x, y in self.valloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss_total += loss.item() * y.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)

        val_loss = loss_total / total
        val_acc = correct / total
        print(f"[Validation] Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save model on final round (optional: only client 1 saves)
        if "round" in config and config["round"] == 5:
            if self.client_id == 1:  # Change condition as needed
                save_path = f"client{self.client_id}_final_model.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ Model saved to: {save_path}")

        return val_loss, total, {"accuracy": val_acc}

# Load data
def load_data(client_id):
    client_dir = f"clients/client{client_id}"
    train_csv = os.path.join(client_dir, "data.csv")
    train_imgs = os.path.join(client_dir, "images")
    val_csv = os.path.join("val", "val.csv")
    val_imgs = os.path.join("val", "images")

    trainset = PlantDataset(train_csv, train_imgs)
    valset = PlantDataset(val_csv, val_imgs)

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    valloader = DataLoader(valset, batch_size=16, shuffle=False)
    return trainloader, valloader

# Run client, port 8080
if __name__ == "__main__":
    client_id = int(sys.argv[1])
    model = create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=True, num_classes=NUM_CLASSES)
    trainloader, valloader = load_data(client_id)
    client = FlowerClient(model, trainloader, valloader, client_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
