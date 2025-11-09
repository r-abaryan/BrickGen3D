"""
Text Encoder Module
====================
This module handles converting text descriptions into numerical embeddings
that our 3D generator can understand.

We use sentence-transformers (a lightweight model) to encode text into 
384-dimensional vectors. This is much simpler than training our own text encoder.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """
    Encodes text descriptions into fixed-size embeddings.
    
    Why sentence-transformers?
    - Pre-trained on large text datasets
    - Lightweight (MiniLM model is only ~80MB)
    - Produces semantically meaningful embeddings
    - No training required!
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the text encoder.
        
        Args:
            model_name: Pre-trained model name. Default 'all-MiniLM-L6-v2' is:
                       - Small (80MB)
                       - Fast
                       - Output dimension: 384
        """
        print(f"Loading text encoder: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Text encoder loaded! Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts):
        """
        Convert text(s) to embeddings.
        
        Args:
            texts: String or list of strings
            
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings
    
    def get_embedding_dim(self):
        """Return the dimension of text embeddings."""
        return self.embedding_dim


# Example usage and testing
if __name__ == "__main__":
    # Test the encoder
    encoder = TextEncoder()
    
    # Test with different descriptions
    test_texts = [
        "a red cube",
        "a blue sphere",
        "a small pyramid"
    ]
    
    embeddings = encoder.encode(test_texts)
    print(f"\nEncoded {len(test_texts)} texts")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 10 dims): {embeddings[0][:10]}")

