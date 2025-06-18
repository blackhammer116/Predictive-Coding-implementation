import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


class PCN(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=256, hidden_dim2=512, hidden_dim3=256, output_dim=10):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.output_dim = output_dim

        # Latent damping parameters
        self.L1 = nn.Parameter(torch.randn(self.hidden_dim1, self.hidden_dim1) * 0.1)
        self.L2 = nn.Parameter(torch.randn(self.hidden_dim2, self.hidden_dim2) * 0.1)
        self.L3 = nn.Parameter(torch.randn(self.hidden_dim3, self.hidden_dim3) * 0.1)
        self.L4 = nn.Parameter(torch.randn(self.output_dim, self.output_dim) * 0.1)
        
        # Weight parameters
        self.W1 = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim1) * 0.1)
        self.W2 = nn.Parameter(torch.randn(self.hidden_dim1, self.hidden_dim2) * 0.1)
        self.W3 = nn.Parameter(torch.randn(self.hidden_dim2, self.hidden_dim3) * 0.1)
        self.W4 = nn.Parameter(torch.randn(self.hidden_dim3, self.output_dim) * 0.1)
        
        # Bias parameters
        self.b1 = nn.Parameter(torch.zeros(self.hidden_dim1))
        self.b2 = nn.Parameter(torch.zeros(self.hidden_dim2))
        self.b3 = nn.Parameter(torch.zeros(self.hidden_dim3))
        self.b4 = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, target=None, n_inference_steps=50, lr_infer=0.001, lr_weight=0.001, update_weights=False):
        batch_size = x.size(0)

        # Initialize latent variables (states)
        # x1 = torch.randn(batch_size, self.hidden_dim1, device=x.device) * 0.1
        # x2 = torch.randn(batch_size, self.hidden_dim2, device=x.device) * 0.1
        # x3 = torch.randn(batch_size, self.hidden_dim3, device=x.device) * 0.1
        # x4 = torch.randn(batch_size, self.output_dim, device=x.device) * 0.1

        # x1 = torch.zeros(batch_size, self.hidden_dim1, device=x.device, requires_grad=True)
        # x2 = torch.zeros(batch_size, self.hidden_dim2, device=x.device, requires_grad=True)
        # x3 = torch.zeros(batch_size, self.hidden_dim3, device=x.device, requires_grad=True)
        # x4 = torch.zeros(batch_size, self.output_dim, device=x.device, requires_grad=True)

        x1 = torch.zeros(batch_size, self.hidden_dim1, device=x.device)
        x2 = torch.zeros(batch_size, self.hidden_dim2, device=x.device)
        x3 = torch.zeros(batch_size, self.hidden_dim3, device=x.device)
        x4 = torch.zeros(batch_size, self.output_dim, device=x.device)

        # Clamp input
        x_clamped = x.view(batch_size, -1).detach()

        if update_weights and target is not None:
            # Clamp top-layer latent variable to target one-hot encoding
            x4 = torch.zeros(batch_size, self.output_dim, device=x.device)
            x4.scatter_(1, target.unsqueeze(1), 1.0)

        for step in range(n_inference_steps):
            # Top-down predictions
            x4_pred = x3 @ self.W4 + self.b4
            x3_pred = self.dropout(self.activation(x2 @ self.W3 + self.b3))
            x2_pred = self.dropout(self.activation(x1 @ self.W2 + self.b2))
            x1_pred = self.dropout(self.activation(x @ self.W1 + self.b1))

            # Bottom-up predictions
            # x3_bu_pred = self.activation(x2 @ self.W3 + self.b3)
            # x2_bu_pred = self.activation(x1 @ self.W2 + self.b2)
            # x1_bu_pred = self.activation(x @ self.W1 + self.b1)

            # Errors (prediction - current state)
            eps4 = x4_pred - x4
            eps3 = x3_pred - x3
            eps2 = x2_pred - x2
            eps1 = x1_pred - x1

            # Lateral damping regularization
            gamma = 0.1
            x3_grad = -eps3 + eps4 @ self.W4.T - gamma * (x3 @ self.L3)
            x2_grad = -eps2 + eps3 @ self.W3.T - gamma * (x2 @ self.L2)
            x1_grad = -eps1 + eps2 @ self.W2.T - gamma * (x1 @ self.L1)

            # Update latent states
            x3 = x3 - lr_infer * x3_grad
            x2 = x2 - lr_infer * x2_grad
            x1 = x1 - lr_infer * x1_grad
            # x4 remains clamped if target is given

            # if step % 50 == 0: 
            #     print(f"  Inference Step {step}:")
            #     print(f"    x1 norm: {torch.norm(x1).item():.4f}, x2 norm: {torch.norm(x2).item():.4f}, x3 norm: {torch.norm(x3).item():.4f}, x4 norm: {torch.norm(x4).item():.4f}")
            #     print(f"    eps1 norm: {torch.norm(eps1).item():.4f}, eps2 norm: {torch.norm(eps2).item():.4f}, eps3 norm: {torch.norm(eps3).item():.4f}, eps4 norm: {torch.norm(eps4).item():.4f}")
            #     print(f"    x1_grad norm: {torch.norm(x1_grad).item():.4f}, x2_grad norm: {torch.norm(x2_grad).item():.4f}, x3_grad norm: {torch.norm(x3_grad).item():.4f}")

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

        output = x3 @ self.W4 + self.b4
        # print(f"  Model Output norm: {torch.norm(output).item():.4f}")
        return output


def train_pcn(model, train_loader, n_epochs=5):
    model.train()
    losses = []
    accuracies = []
    optimizer = optim.Adam(pcn.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(n_epochs):
        total_loss = 0.0
        correct = 0
        tot = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}', leave=False)
        for batch_idx, (data, target) in enumerate(progress_bar):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            output = model(data, target=target, update_weights=True)

            target_onehot = F.one_hot(target, num_classes=output.size(1)).float()
            loss = ((output - target_onehot) ** 2).mean()
            # loss = F.cross_entropy(output, target)
            total_loss += loss.item()

            # Calculate accuracy for the current batch
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            tot += target.size(0)

            # Update the progress bar description with current loss and accuracy
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': correct / tot})

        # Calculate average loss and accuracy for the epoch
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / tot

        scheduler.step()
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss:.6f}, Accuracy: {epoch_accuracy:.4f}")

    plt.plot(losses, label='Loss')
    plt.plot(accuracies, label='Accuracy')
    plt.legend()
    plt.show()


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
hidden_dim = 512
hidden_dim2 = 1024
hidden_dim3 = 512
output_dim = 10

pcn = PCN(input_dim, hidden_dim, hidden_dim2, hidden_dim3, output_dim).to(device)

train_pcn(pcn, train_loader, n_epochs=10)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_pcn(pcn, test_loader)