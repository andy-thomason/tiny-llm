# tiny-llm
Experiments in minimal LLM construction.

A simple character-level language model with transformer architecture, including self-attention mechanism and training on Shakespeare text.

## Features

- **Character-level tokenizer**: Maps text to character indices and back
- **Self-attention mechanism**: Multi-head attention for capturing dependencies
- **Transformer architecture**: Includes positional embeddings, layer normalization, and feed-forward networks
- **Training pipeline**: Complete training loop with loss estimation
- **Text generation**: Sampling-based text generation with temperature control
- **Pre-trained on Shakespeare**: Trained on the tiny Shakespeare dataset

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on the Shakespeare dataset:

```bash
python train.py
```

This will:
1. Download the tiny Shakespeare dataset (~1MB)
2. Build a character-level vocabulary
3. Train a transformer model with self-attention
4. Save the trained model to `tiny_llm_model.pt`

Training takes approximately 5-10 minutes on CPU, faster on GPU.

### Text Generation

Generate text using the trained model:

```bash
python generate.py
```

This will load the trained model and allow you to:
- Enter prompts for text generation
- Generate text with customizable length and temperature
- Explore the model's learned patterns

## Model Architecture

The model consists of:
- Token embedding layer (vocabulary → embedding space)
- Position embedding layer (positional encoding)
- 4 transformer blocks, each with:
  - Multi-head self-attention (4 heads)
  - Feed-forward network
  - Layer normalization
  - Residual connections
- Output layer (embedding → vocabulary)

Default hyperparameters:
- Embedding dimension: 64
- Number of heads: 4
- Number of layers: 4
- Block size (context length): 128
- Dropout: 0.1

## Files

- `tokenizer.py`: Character-level tokenizer implementation
- `model.py`: Transformer model with self-attention
- `train.py`: Training script
- `generate.py`: Text generation script
- `data_utils.py`: Data downloading and loading utilities
- `requirements.txt`: Python dependencies

## Example Output

After training, the model can generate Shakespeare-like text:

```
DUKE OF YORK:
What say you, lords? Will you go with me?

KING RICHARD III:
I will, my lord, and hope to see the day
When I shall live to see thee in thy grave.
```

## License

This is an experimental educational project.
