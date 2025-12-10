"""
Demo script - quick training for demonstration purposes.
"""

import torch
from tokenizer import CharTokenizer
from model import TinyLLM
from data_utils import get_shakespeare_data, load_text

# Reduced hyperparameters for quick demo
BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_ITERS = 1000  # Reduced from 5000
EVAL_INTERVAL = 200
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 100  # Reduced from 200
N_EMBD = 64
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1


def get_batch(data, batch_size, block_size):
    """Generate a random batch of data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters):
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def main():
    """Main demo function."""
    print(f"Using device: {DEVICE}")
    
    # Download and load data
    print("\n=== Loading Data ===")
    data_path = get_shakespeare_data()
    text = load_text(data_path)
    print(f"Loaded {len(text)} characters")
    
    # Create tokenizer
    print("\n=== Building Tokenizer ===")
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Encode text and split into train/val
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=N_EMBD,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT
    )
    model = model.to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n=== Training (Quick Demo - 1000 iterations) ===")
    for iter in range(MAX_ITERS):
        # Evaluate loss periodically
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, BATCH_SIZE, BLOCK_SIZE, EVAL_ITERS)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Get batch and compute loss
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        logits, loss = model(xb, yb)
        
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Generate sample text
    print("\n=== Generating Text (Sample 1 - Random) ===")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=300, temperature=0.8)
    generated_text = tokenizer.decode(generated[0].tolist())
    print(generated_text)
    
    print("\n=== Generating Text (Sample 2 - With Prompt) ===")
    prompt = "ROMEO:"
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=300, temperature=0.8)
    generated_text = tokenizer.decode(generated[0].tolist())
    print(generated_text)
    
    # Save model
    print("\n=== Saving Model ===")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_char_to_idx': tokenizer.char_to_idx,
        'tokenizer_idx_to_char': tokenizer.idx_to_char,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'n_embd': N_EMBD,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'block_size': BLOCK_SIZE,
            'dropout': DROPOUT,
        }
    }, 'tiny_llm_demo.pt')
    print("Model saved to tiny_llm_demo.pt")


if __name__ == '__main__':
    main()
