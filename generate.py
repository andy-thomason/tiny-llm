"""
Text generation script using trained tiny-llm model.
"""

import torch
from tokenizer import CharTokenizer
from model import TinyLLM


def load_model(model_path='tiny_llm_model.pt', device='cpu'):
    """
    Load trained model and tokenizer.
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate tokenizer
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = checkpoint['tokenizer_char_to_idx']
    tokenizer.idx_to_char = checkpoint['tokenizer_idx_to_char']
    tokenizer.vocab_size = checkpoint['config']['vocab_size']
    
    # Recreate model
    model = TinyLLM(
        vocab_size=checkpoint['config']['vocab_size'],
        n_embd=checkpoint['config']['n_embd'],
        num_heads=checkpoint['config']['num_heads'],
        num_layers=checkpoint['config']['num_layers'],
        block_size=checkpoint['config']['block_size'],
        dropout=checkpoint['config']['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt='', max_tokens=500, temperature=0.8, device='cpu'):
    """
    Generate text from a prompt.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer
        prompt: Initial text prompt
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature
        device: Device to run on
        
    Returns:
        Generated text string
    """
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
    return tokenizer.decode(generated[0].tolist())


def main():
    """Main inference function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model('tiny_llm_model.pt', device)
    print(f"Model loaded. Vocabulary size: {tokenizer.vocab_size}")
    
    # Interactive generation
    print("\n=== Text Generation ===")
    print("Enter a prompt (or press Enter for random generation):")
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == 'quit':
            break
        
        print("\nGenerating...")
        generated = generate_text(model, tokenizer, prompt, max_tokens=300, temperature=0.8, device=device)
        print(f"\n{generated}\n")
        print("-" * 80)


if __name__ == '__main__':
    main()
