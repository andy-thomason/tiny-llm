"""
Simple character-level tokenizer for tiny-llm.
"""

class CharTokenizer:
    """Character-level tokenizer that maps characters to integers."""
    
    def __init__(self, text=None):
        """
        Initialize tokenizer with optional training text.
        
        Args:
            text: Optional text string to build vocabulary from
        """
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        if text is not None:
            self.build_vocab(text)
    
    def build_vocab(self, text):
        """
        Build vocabulary from text.
        
        Args:
            text: String to extract unique characters from
        """
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        """
        Encode text to list of token indices.
        
        Args:
            text: String to encode
            
        Returns:
            List of integer token indices
        """
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        """
        Decode list of token indices to text.
        
        Args:
            indices: List of integer token indices
            
        Returns:
            Decoded string
        """
        return ''.join([self.idx_to_char[i] for i in indices])
