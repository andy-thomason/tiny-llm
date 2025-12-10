"""
Utilities for downloading and preparing training data.
"""

import os
import requests


def download_text_corpus(url, save_path):
    """
    Download a text corpus from URL.
    
    Args:
        url: URL to download from
        save_path: Path to save the downloaded file
        
    Returns:
        Path to downloaded file
    """
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}")
        return save_path
    
    print(f"Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    print(f"Downloaded to {save_path}")
    return save_path


def get_shakespeare_data(data_dir='data'):
    """
    Download tiny Shakespeare dataset.
    
    Args:
        data_dir: Directory to save data
        
    Returns:
        Path to downloaded file
    """
    os.makedirs(data_dir, exist_ok=True)
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    save_path = os.path.join(data_dir, 'shakespeare.txt')
    return download_text_corpus(url, save_path)


def load_text(file_path):
    """
    Load text from file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Text string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
