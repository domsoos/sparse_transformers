import torch
import torch.nn.functional as F
from torch import nn

class SparseAttention(nn.Module):
    def __init__(self, embed_size, num_heads, sparsity_pattern):
        super(SparseAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.sparsity_pattern = sparsity_pattern  # Define your sparsity pattern

    def forward(self, x):
        # Splitting the embedding size into multiple heads
        B, N, E = x.shape
        q = self.queries(x).view(B, N, self.num_heads, E // self.num_heads)
        k = self.keys(x).view(B, N, self.num_heads, E // self.num_heads)
        v = self.values(x).view(B, N, self.num_heads, E // self.num_heads)
        
        # Apply sparsity pattern to the attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.sparsity_pattern
        attention = F.softmax(attention_scores, dim=-1)
        
        # Attend to the values
        out = torch.matmul(attention, v).view(B, N, E)
        return out
