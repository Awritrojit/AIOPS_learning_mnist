import os

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

import mlflow.pytorch
from mlflow import MlflowClient

# Set random seed for reproducibility
torch.manual_seed(42)

# Define a simple MLP model
class SimpleMLP(L.LightningModule):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Create data loaders
batch_size = 64

# Initialize the model, loss function, and optimizer
model = SimpleMLP().to(device)


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")



# Load MNIST dataset.
train_ds = MNIST(
    os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_ds, batch_size=batch_size)

# Initialize a trainer.
trainer = L.Trainer(max_epochs=5)

# Auto log all MLflow entities
mlflow.pytorch.autolog()

# Train the model.
with mlflow.start_run() as run:
    trainer.fit(model, train_loader)

# Fetch the auto logged parameters and metrics.
print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

