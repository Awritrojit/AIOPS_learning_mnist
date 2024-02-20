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

        self.test_step_outputs_loss = []
        self.test_step_outputs_acc = []

        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.test_step_outputs_loss.append(loss)

        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)
        self.test_step_outputs_acc.append(acc)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs_loss).mean()
        avg_acc = torch.stack(self.test_step_outputs_acc).mean()

        # Log average test loss and accuracy for the entire test set.
        self.log("avg_test_loss", avg_loss)
        self.log("avg_test_acc", avg_acc)

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
mnist_dataset = MNIST(
    os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
)

# Split the dataset into training and test sets
train_size = int(0.8 * len(mnist_dataset))
test_size = len(mnist_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(mnist_dataset, [train_size, test_size])

# Save the test set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Create a directory to save PNG images
save_dir = "./data/mnist_test_images"
os.makedirs(save_dir, exist_ok=True)

# Iterate through the test set, convert tensors to images, and save as PNG
for i, (image, label) in enumerate(zip(*next(iter(test_loader)))):
    image = transforms.ToPILImage()(image)
    file_name = os.path.join(save_dir, f"{label.item()}_image_{i}.png")
    image.save(file_name)

# Initialize a trainer.
trainer = L.Trainer(max_epochs=5)

# Auto log all MLflow entities
mlflow.pytorch.autolog()

# Train the model.
with mlflow.start_run() as run:
    trainer.fit(model, train_loader)

    trainer.test(model, test_loader)

# Fetch the auto logged parameters and metrics.
print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

