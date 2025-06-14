import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCN(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.L1 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.1)
        self.L2 = nn.Parameter(torch.randn(self.output_dim, self.output_dim) * 0.1)

        self.W1 = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim) * 0.1)
        self.W2 = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim) * 0.1)

        self.b1 = nn.Parameter(torch.zeros(self.hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Sigmoid()
    
    def forward(self, x, target=None, n_inference_steps=20, lr_infer=0.1, lr_weight=0.001, update_weights=False):
        batch_size = x.size(0)

        x1 = torch.randn(batch_size, self.W1.shape[1], device=x.device) * 0.1
        x2 = torch.randn(batch_size, self.W2.shape[1], device=x.device) * 0.1

        for _ in range(n_inference_steps):
            x1_pred = self.activation(x @ self.W1 + self.b1)
            x2_pred = self.activation(x1 @ self.W2 + self.b2)

            eps1 = x1 - x1_pred
            eps2 = x2 - x2_pred

            x2_grad = eps2 + x2 @ self.L2
            x1_grad = eps1 - (x2_grad @ self.W2.T) + x1 @ self.L1

            x1 = x1 - lr_infer * x1_grad
            x2 = x2 - lr_infer * x2_grad

        if update_weights and target is not None:
            target_onehot = torch.zeros_like(x2)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

            eps2 = x2 - target_onehot
            eps1 = x1 - self.activation(x @ self.W1 + self.b1)

            with torch.no_grad():
                self.W2 -= lr_weight * (x1.T @ eps2) / batch_size
                self.b2 -= lr_weight * eps2.mean(dim=0)

                self.W1 -= lr_weight * (x.T @ eps1) / batch_size
                self.b1 -= lr_weight * eps1.mean(dim=0)
        
        return x2
        


def train_pcn(model, train_loader, epoch=5, learning_rate_w = 0.001):
    # optimizer = optim.SGD(model.parameters(), lr = learning_rate_w)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch):
        total_loss = 0.0

        for batch_idx , (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            data, target = data.to(device), target.to(device)

            output = model(data, target, update_weights=True)
            target_onehot = torch.zeros_like(output)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            loss = ((output - target_onehot) ** 2).mean()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

input_dim = 784
hidden_dim = 256
output_dim = 10

pcn = PCN(input_dim, hidden_dim, output_dim).to(device)

train_pcn(pcn, train_loader, epoch=5)

def test_pcn(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            data, target = data.to(device), target.to(device)

            output = model(data, n_inference_steps=20)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_pcn(pcn, test_loader)