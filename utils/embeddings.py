"""
Embeddings Module
Generates embeddings using Hugging Face sentence-transformers models
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import logging
import time
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using Hugging Face models"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize embedding generator

        Args:
            model_name: Hugging Face model name for embeddings
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise

    @lru_cache(maxsize=128)
    def _get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text (cached)"""
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding generation")
            return np.zeros(self.model.get_sentence_embedding_dimension())

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.model.get_sentence_embedding_dimension())

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more texts

        Args:
            texts: Single text string or list of text strings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            return self._get_single_embedding(texts)
        elif isinstance(texts, list):
            if len(texts) == 0:
                return np.array([])

            # Process in batches for efficiency
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {str(e)}")
                # Fallback to individual embeddings
                return np.array([self._get_single_embedding(text) for text in texts])
        else:
            raise TypeError(f"Expected str or List[str], got {type(texts)}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.model.get_sentence_embedding_dimension()

    def update_model(self, model_name: str):
        """Update to a different model"""
        if model_name != self.model_name:
            self.model_name = model_name
            self._load_model()
            # Clear cache since model changed
            self._get_single_embedding.cache_clear()
