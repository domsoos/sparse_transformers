import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18  # Using ResNet18 as a feature extractor
import numpy as np
import math

class SparseAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SparseAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, "Embed size must be divisible by num_heads"

        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        
        self.feature_projection = nn.Parameter(torch.randn(self.head_dim, self.head_dim))

    def forward(self, x):
        B, N, E = x.shape
        q = self.queries(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.keys(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.values(x).reshape(B, N, self.num_heads, self.head_dim)

        # Projecting to feature space
        q = torch.einsum('bnhe,ei->bnhi', q, self.feature_projection)
        k = torch.einsum('bnhe,ei->bnhi', k, self.feature_projection)

        # Compute attention scores
        scores = torch.einsum('bnhi,bmhi->bnhm', q, k) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.einsum('bnhm,bmhe->bnhe', attention, v).reshape(B, N, E)
        return out



class CIFARSparseAttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, num_classes=10):
        super().__init__()
        # Feature Extractor
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Project extracted features to the required embedding size
        self.feature_projection = nn.Linear(512, embed_size)

        # Sparse Attention Mechanism
        self.sparse_attention = EfficientSparseAttention(embed_size, num_heads)

        # Classifier Head
        self.classifier = nn.Linear(embed_size, num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Extract features from the image
        features = self.feature_extractor(x)  # Shape: [B, 512]
        features = self.feature_projection(features)  # Shape: [B, embed_size]
        
        # Apply sparse attention
        # Assuming we can treat the features as a sequence here; otherwise, reshape might be necessary
        attention_output = self.sparse_attention(features.unsqueeze(1)).squeeze(1)  # Shape: [B, embed_size]
        
        # Classify
        logits = self.classifier(attention_output)
        return logits

# CIFAR-10 Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit ResNet18 input
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(cifar_train, batch_size=32, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size=32, shuffle=False)

# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")
model = CIFARSparseAttentionModel(embed_size=512, num_heads=8).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Training and Validation
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')

def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

# Train and Evaluate the Model
epochs = 10
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    validate(model, device, test_loader)
