"""
Simple tests to verify the tiny-llm components work correctly.
"""

import torch
from tokenizer import CharTokenizer
from model import TinyLLM


def test_tokenizer():
    """Test character tokenizer."""
    print("Testing tokenizer...")
    
    # Test text
    text = "Hello, World!"
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    
    # Test encoding
    encoded = tokenizer.encode(text)
    assert isinstance(encoded, list), "Encode should return a list"
    assert len(encoded) == len(text), "Encoded length should match text length"
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"Decoded text should match original. Got: {decoded}"
    
    print(f"✓ Tokenizer works correctly")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Sample encoding: '{text}' -> {encoded[:5]}...")
    

def test_model():
    """Test model forward pass."""
    print("\nTesting model...")
    
    # Small model for testing
    vocab_size = 50
    batch_size = 2
    block_size = 16
    
    # Create model
    model = TinyLLM(
        vocab_size=vocab_size,
        n_embd=32,
        num_heads=2,
        num_layers=2,
        block_size=block_size,
        dropout=0.0
    )
    
    # Create random input
    x = torch.randint(0, vocab_size, (batch_size, block_size))
    
    # Forward pass without targets
    logits, loss = model(x)
    assert logits.shape == (batch_size, block_size, vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert loss is None, "Loss should be None when targets not provided"
    
    # Forward pass with targets
    y = torch.randint(0, vocab_size, (batch_size, block_size))
    logits, loss = model(x, y)
    assert loss is not None, "Loss should be computed when targets provided"
    assert isinstance(loss.item(), float), "Loss should be a scalar"
    
    print(f"✓ Model forward pass works correctly")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss value: {loss.item():.4f}")


def test_generation():
    """Test text generation."""
    print("\nTesting generation...")
    
    # Small model for testing
    vocab_size = 50
    block_size = 16
    
    model = TinyLLM(
        vocab_size=vocab_size,
        n_embd=32,
        num_heads=2,
        num_layers=2,
        block_size=block_size,
        dropout=0.0
    )
    model.eval()
    
    # Generate from empty context
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=10, temperature=1.0)
    
    assert generated.shape[0] == 1, "Batch size should be 1"
    assert generated.shape[1] == 11, f"Generated sequence should have 11 tokens (1 + 10), got {generated.shape[1]}"
    assert torch.all((generated >= 0) & (generated < vocab_size)), "Generated tokens should be in valid range"
    
    print(f"✓ Generation works correctly")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample tokens: {generated[0, :5].tolist()}")


def test_attention():
    """Test attention mechanism."""
    print("\nTesting attention mechanism...")
    
    from model import Head, MultiHeadAttention
    
    # Test single head
    n_embd = 32
    head_size = 8
    block_size = 16
    batch_size = 2
    
    head = Head(n_embd, head_size, block_size)
    x = torch.randn(batch_size, block_size, n_embd)
    out = head(x)
    
    assert out.shape == (batch_size, block_size, head_size), f"Unexpected head output shape: {out.shape}"
    
    # Test multi-head attention
    num_heads = 4
    mha = MultiHeadAttention(n_embd, num_heads, head_size, block_size)
    out = mha(x)
    
    assert out.shape == (batch_size, block_size, n_embd), f"Unexpected MHA output shape: {out.shape}"
    
    print(f"✓ Attention mechanism works correctly")
    print(f"  Single head output: {out.shape}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Running tiny-llm tests")
    print("=" * 50)
    
    test_tokenizer()
    test_model()
    test_generation()
    test_attention()
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
