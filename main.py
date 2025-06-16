import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCN(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=256, hidden_dim2=1024, hidden_dim3=256, output_dim=10):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.output_dim = output_dim

        self.L1 = nn.Parameter(torch.randn(self.hidden_dim1, self.hidden_dim1) * 0.1)
        self.L2 = nn.Parameter(torch.randn(self.hidden_dim2, self.hidden_dim2) * 0.1)
        self.L3 = nn.Parameter(torch.randn(self.hidden_dim3, self.hidden_dim3) * 0.1)
        self.L4 = nn.Parameter(torch.randn(self.output_dim, self.output_dim) * 0.1)

        self.W1 = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim1) * 0.1)
        self.W2 = nn.Parameter(torch.randn(self.hidden_dim1, self.hidden_dim2) * 0.1)
        self.W3 = nn.Parameter(torch.randn(self.hidden_dim2, self.hidden_dim3) * 0.1)
        self.W4 = nn.Parameter(torch.randn(self.hidden_dim3, self.output_dim) * 0.1)
        # self.W2 = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim) * 0.1)

        self.b1 = nn.Parameter(torch.zeros(self.hidden_dim1))
        self.b2 = nn.Parameter(torch.zeros(self.hidden_dim2))
        self.b3 = nn.Parameter(torch.zeros(self.hidden_dim3))
        self.b4 = nn.Parameter(torch.zeros(self.output_dim))

        # self.b2 = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Sigmoid()
    
    def forward(self, x, target=None, n_inference_steps=20, lr_infer=0.1, lr_weight=0.001, update_weights=False):
        batch_size = x.size(0)

        # Initialize latent variables (states)
        x1 = torch.randn(batch_size, self.hidden_dim1, device=x.device) * 0.1
        x2 = torch.randn(batch_size, self.hidden_dim2, device=x.device) * 0.1
        x3 = torch.randn(batch_size, self.hidden_dim3, device=x.device) * 0.1
        x4 = torch.randn(batch_size, self.output_dim, device=x.device) * 0.1

        if update_weights and target is not None:
            # Clamp top-layer latent variable to target one-hot encoding
            x4 = torch.zeros(batch_size, self.output_dim, device=x.device)
            x4.scatter_(1, target.unsqueeze(1), 1.0)

        for _ in range(n_inference_steps):
            # Top-down predictions
            x3_pred = self.activation(x2 @ self.W3 + self.b3)
            x2_pred = self.activation(x1 @ self.W2 + self.b2)
            x1_pred = self.activation(x @ self.W1 + self.b1)

            # Bottom-up predictions
            x4_pred = x3 @ self.W4 + self.b4
            x3_bu_pred = self.activation(x2 @ self.W3 + self.b3)
            x2_bu_pred = self.activation(x1 @ self.W2 + self.b2)
            x1_bu_pred = self.activation(x @ self.W1 + self.b1)

            # Errors (prediction - current state)
            eps4 = x4_pred - x4
            eps3 = x3_pred - x3
            eps2 = x2_pred - x2
            eps1 = x1_pred - x1

            # Lateral damping (optional regularization)
            gamma = 0.1
            x3_grad = -eps3 + eps4 @ self.W4.T - gamma * (x3 @ self.L3)
            x2_grad = -eps2 + eps3 @ self.W3.T - gamma * (x2 @ self.L2)
            x1_grad = -eps1 + eps2 @ self.W2.T - gamma * (x1 @ self.L1)

            # Update latent states
            x3 = x3 - lr_infer * x3_grad
            x2 = x2 - lr_infer * x2_grad
            x1 = x1 - lr_infer * x1_grad
            # x4 remains clamped if target is given

        # Weight updates after inference
        if update_weights and target is not None:
            with torch.no_grad():
                eps4 = (x3 @ self.W4 + self.b4) - x4
                eps3 = self.activation(x2 @ self.W3 + self.b3) - x3
                eps2 = self.activation(x1 @ self.W2 + self.b2) - x2
                eps1 = self.activation(x @ self.W1 + self.b1) - x1

                self.W4 -= lr_weight * (x3.T @ eps4) / batch_size
                self.b4 -= lr_weight * eps4.mean(dim=0)

                self.W3 -= lr_weight * (x2.T @ eps3) / batch_size
                self.b3 -= lr_weight * eps3.mean(dim=0)

                self.W2 -= lr_weight * (x1.T @ eps2) / batch_size
                self.b2 -= lr_weight * eps2.mean(dim=0)

                self.W1 -= lr_weight * (x.T @ eps1) / batch_size
                self.b1 -= lr_weight * eps1.mean(dim=0)

        # return x4
        return  x3 @ self.W4 + self.b4


def train_pcn(model, train_loader, n_epochs=5):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            output = model(data, target=target, update_weights=True)

            # Optional: track MSE loss for monitoring
            target_onehot = F.one_hot(target, num_classes=output.size(1)).float()
            loss = ((output - target_onehot) ** 2).mean()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(train_loader):.4f}")


def test_pcn(model, test_loader, n_inference_steps=20):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            output = model(data, n_inference_steps=n_inference_steps, update_weights=False)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

    print(f"Test Accuracy: {100.0 * correct / total:.2f}%")


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

input_dim = 784
hidden_dim = 256
hidden_dim2 = 1024
hidden_dim3 = 256
output_dim = 10

pcn = PCN(input_dim, hidden_dim, hidden_dim2, hidden_dim3, output_dim).to(device)

train_pcn(pcn, train_loader, n_epochs=5)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_pcn(pcn, test_loader)