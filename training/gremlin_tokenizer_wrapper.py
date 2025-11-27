"""
GREMLIN Tokenizer Wrapper

SECURITY COMPLIANCE:
- Zero transformers/hub dependencies
- Pure Python with tokenizers library only
- Local file operations only
- No network calls, no telemetry

Wraps the custom GREMLIN tokenizer (128K vocab) for use with training pipeline.

Author: GREMLIN Team
License: MIT
"""

import torch
from tokenizers import Tokenizer
from pathlib import Path
from typing import List, Dict, Union, Optional


class GremlinTokenizer:
    """
    Wrapper for GREMLIN's custom 128K BPE tokenizer.

    Provides encode/decode functionality compatible with training pipeline.
    No transformers library dependency.
    """

    def __init__(self, tokenizer_path: str):
        """
        Initialize GREMLIN tokenizer.

        Args:
            tokenizer_path: Path to gremlin_tokenizer.json
        """
        self.tokenizer_path = Path(tokenizer_path)

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")

        # Load tokenizer from JSON
        print(f"📚 Loading GREMLIN tokenizer from {self.tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        # Get vocabulary
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)

        # Special tokens (from GREMLIN tokenizer training)
        self.special_tokens = {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "unk_token": "<|unk|>",
            "pad_token": "<|padding|>",
            "gremlin_token": "<|gremlin|>",
        }

        # Get special token IDs
        self.bos_token_id = self.vocab.get(self.special_tokens["bos_token"], 0)
        self.eos_token_id = self.vocab.get(self.special_tokens["eos_token"], 0)
        self.unk_token_id = self.vocab.get(self.special_tokens["unk_token"], 3)
        self.pad_token_id = self.vocab.get(self.special_tokens["pad_token"], 1)
        self.gremlin_token_id = self.vocab.get(self.special_tokens["gremlin_token"], 2)

        print(f"✓ Tokenizer loaded")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Special tokens: {list(self.special_tokens.keys())}")
        print(f"  BOS/EOS ID: {self.bos_token_id}")
        print(f"  PAD ID: {self.pad_token_id}")
        print(f"  UNK ID: {self.unk_token_id}")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: 'pt' for PyTorch tensor, None for list
            padding: Whether to pad to max_length
            max_length: Maximum sequence length
            truncation: Whether to truncate to max_length

        Returns:
            Token IDs as list or tensor
        """
        # Encode text
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids

        # Add special tokens if requested
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        # Truncate if needed
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Pad if needed
        if padding and max_length is not None and len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))

        # Convert to tensor if requested
        if return_tensors == "pt":
            return torch.tensor([token_ids], dtype=torch.long)
        else:
            return token_ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs (list or tensor)
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Handle batch dimension
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        # Decode
        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        return text

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple texts as a batch.

        Args:
            texts: List of input texts
            add_special_tokens: Whether to add BOS/EOS tokens
            padding: Whether to pad to max_length
            max_length: Maximum sequence length
            truncation: Whether to truncate to max_length
            return_tensors: 'pt' for PyTorch tensors

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Encode all texts
        all_token_ids = []
        for text in texts:
            token_ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors=None,
                padding=False,
                truncation=truncation,
                max_length=max_length,
            )
            all_token_ids.append(token_ids)

        # Find max length if not specified
        if max_length is None:
            max_length = max(len(ids) for ids in all_token_ids)

        # Pad all sequences
        input_ids = []
        attention_masks = []

        for token_ids in all_token_ids:
            # Truncate if needed
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(token_ids)

            # Pad if needed
            if len(token_ids) < max_length:
                padding_length = max_length - len(token_ids)
                token_ids = token_ids + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            input_ids.append(token_ids)
            attention_masks.append(attention_mask)

        # Convert to tensors
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

        return result

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode multiple sequences.

        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Decode each sequence
        texts = [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in token_ids
        ]

        return texts

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.vocab

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Callable interface for easy tokenization.

        Args:
            text: Single text or list of texts
            add_special_tokens: Whether to add BOS/EOS tokens
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: 'pt' for PyTorch tensors

        Returns:
            Token IDs (single) or dictionary with input_ids and attention_mask (batch)
        """
        if isinstance(text, str):
            # Single text
            return self.encode(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
            )
        else:
            # Batch of texts
            return self.batch_encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                return_tensors=return_tensors,
            )


def load_gremlin_tokenizer(tokenizer_path: Optional[str] = None) -> GremlinTokenizer:
    """
    Convenience function to load GREMLIN tokenizer.

    Args:
        tokenizer_path: Path to tokenizer JSON, or None for default location

    Returns:
        Loaded GremlinTokenizer
    """
    if tokenizer_path is None:
        # Default location
        tokenizer_path = "F:/dev/GREMLIN_Claude_Code_Web_track/models/tokenizer/gremlin_tokenizer.json"

    return GremlinTokenizer(tokenizer_path)


if __name__ == "__main__":
    # Test tokenizer
    print("Testing GREMLIN tokenizer wrapper...")
    print("=" * 70)

    # Load tokenizer
    tokenizer = load_gremlin_tokenizer()

    # Test encoding
    test_text = "Artificial intelligence is transforming the world."
    print(f"\nTest text: {test_text}")

    token_ids = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Token IDs: {token_ids}")
    print(f"Length: {len(token_ids)} tokens")

    # Test decoding
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Decoded: {decoded}")

    # Test batch encoding
    print("\nTesting batch encoding...")
    batch_texts = [
        "Hello world",
        "GREMLIN is a semantic compression engine",
        "Frontier science!"
    ]

    batch_output = tokenizer.batch_encode(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=50,
    )

    print(f"Batch input_ids shape: {batch_output['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch_output['attention_mask'].shape}")

    # Test batch decoding
    decoded_batch = tokenizer.batch_decode(batch_output['input_ids'])
    print(f"Decoded batch:")
    for i, text in enumerate(decoded_batch):
        print(f"  {i+1}. {text}")

    print("\n✓ Tokenizer test complete")
