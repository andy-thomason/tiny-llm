"""
Simple transformer-based language model with self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Single head of self-attention."""
    
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        """
        Initialize attention head.
        
        Args:
            n_embd: Embedding dimension
            head_size: Size of attention head
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass of attention head.
        
        Args:
            x: Input tensor of shape (batch, time, channels)
            
        Returns:
            Output tensor of shape (batch, time, head_size)
        """
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Weighted aggregation
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            n_embd: Embedding dimension
            num_heads: Number of attention heads
            head_size: Size of each attention head
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass of multi-head attention."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple feed-forward network."""
    
    def __init__(self, n_embd, dropout=0.1):
        """
        Initialize feed-forward network.
        
        Args:
            n_embd: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """Forward pass of feed-forward network."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    
    def __init__(self, n_embd, num_heads, block_size, dropout=0.1):
        """
        Initialize transformer block.
        
        Args:
            n_embd: Embedding dimension
            num_heads: Number of attention heads
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(n_embd, num_heads, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        """Forward pass with residual connections."""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyLLM(nn.Module):
    """Simple transformer-based language model."""
    
    def __init__(self, vocab_size, n_embd=64, num_heads=4, num_layers=4, block_size=128, dropout=0.1):
        """
        Initialize tiny language model.
        
        Args:
            vocab_size: Size of vocabulary
            n_embd: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        Forward pass of the model.
        
        Args:
            idx: Input token indices of shape (batch, time)
            targets: Optional target indices for loss calculation
            
        Returns:
            If targets is None: logits of shape (batch, time, vocab_size)
            If targets is provided: (logits, loss)
        """
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate new tokens from initial context.
        
        Args:
            idx: Initial context indices of shape (batch, time)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated token indices of shape (batch, time + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx
