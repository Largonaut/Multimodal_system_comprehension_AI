"""
Vector Store for GREMLIN
Manages embeddings and concept-word mappings using MRL vectors.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class VectorStore:
    """
    Manages vector embeddings for concepts and synthetic words.

    Uses EmbeddingGemma for MRL (Matryoshka Representation Learning) vectors
    to track word usage and semantic mappings.
    """

    def __init__(
        self,
        embedding_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize vector store.

        Args:
            embedding_model_path: Path to EmbeddingGemma model (None = use defaults)
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        self.embeddings: Dict[str, np.ndarray] = {}
        self.concept_words: Dict[str, List[str]] = {}  # concept_id -> [words]

        if not HAS_TRANSFORMERS:
            print("⚠️  Transformers not installed. Using lightweight fallback mode.")
            self.model = None
            self.tokenizer = None
            return

        # Try to load EmbeddingGemma or fallback to sentence-transformers
        if embedding_model_path and Path(embedding_model_path).exists():
            print(f"📦 Loading EmbeddingGemma from {embedding_model_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
                self.model = AutoModel.from_pretrained(embedding_model_path).to(device)
                self.model.eval()
                print(f"✅ EmbeddingGemma loaded on {device}")
            except Exception as e:
                print(f"⚠️  Could not load EmbeddingGemma: {e}")
                self._load_fallback()
        else:
            self._load_fallback()

    def _load_fallback(self):
        """Load a lightweight fallback embedding model."""
        print("📦 Loading fallback embedding model (all-MiniLM-L6-v2)...")
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print(f"✅ Fallback model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Could not load fallback model: {e}")
            self.model = None
            self.tokenizer = None

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector (numpy array)
        """
        if self.model is None:
            # Fallback: simple hash-based pseudo-embedding
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            pseudo_vec = np.frombuffer(hash_obj.digest(), dtype=np.uint8)[:128]
            return pseudo_vec.astype(np.float32) / 255.0

        # Use model for real embeddings
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)

            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()[0]

    def add_concept(self, concept_id: str, concept_text: str, words: List[str]):
        """
        Add a concept with its synthetic words to the vector store.

        Args:
            concept_id: Unique concept identifier
            concept_text: English text for the concept
            words: List of synthetic words for this concept
        """
        # Generate embedding for the concept
        embedding = self.embed_text(concept_text)
        self.embeddings[concept_id] = embedding
        self.concept_words[concept_id] = words

    def find_concept_for_word(self, word: str) -> Optional[str]:
        """
        Find which concept a synthetic word belongs to.

        Args:
            word: Synthetic word to look up

        Returns:
            concept_id or None
        """
        for concept_id, words in self.concept_words.items():
            if word in words:
                return concept_id
        return None

    def get_concept_embedding(self, concept_id: str) -> Optional[np.ndarray]:
        """Get embedding for a concept."""
        return self.embeddings.get(concept_id)

    def similarity(self, text: str, concept_id: str) -> float:
        """
        Calculate similarity between text and a concept.

        Args:
            text: Input text
            concept_id: Concept to compare to

        Returns:
            Similarity score (0-1)
        """
        concept_emb = self.embeddings.get(concept_id)
        if concept_emb is None:
            return 0.0

        text_emb = self.embed_text(text)

        # Cosine similarity
        dot_product = np.dot(text_emb, concept_emb)
        norm_product = np.linalg.norm(text_emb) * np.linalg.norm(concept_emb)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def save(self, path: Path):
        """Save vector store to disk."""
        data = {
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
            'concept_words': self.concept_words
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Vector store saved to {path}")

    def load(self, path: Path):
        """Load vector store from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.embeddings = {k: np.array(v) for k, v in data['embeddings'].items()}
        self.concept_words = data['concept_words']

        print(f"📂 Vector store loaded from {path}")

    def __repr__(self):
        return f"VectorStore({len(self.embeddings)} concepts, device={self.device})"
