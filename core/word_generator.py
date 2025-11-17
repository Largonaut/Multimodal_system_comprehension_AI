"""
Unicode Word Generator
Generates random Unicode strings for synthetic language words.
"""

import random
import string
from typing import List, Set, Tuple
from dataclasses import dataclass


@dataclass
class UnicodeBlock:
    """Represents a Unicode character block."""
    name: str
    start: int
    end: int

    def generate_char(self) -> str:
        """Generate a random character from this block."""
        return chr(random.randint(self.start, self.end))


class WordGenerator:
    """
    Generates random Unicode strings for synthetic language words.

    Supports all Unicode blocks to create truly unique, random words
    that have no semantic meaning in any real language.
    """

    # Unicode blocks available for word generation
    UNICODE_BLOCKS = {
        'latin_basic': UnicodeBlock('Latin Basic', 0x0041, 0x007A),
        'latin_extended_a': UnicodeBlock('Latin Extended-A', 0x0100, 0x017F),
        'latin_extended_b': UnicodeBlock('Latin Extended-B', 0x0180, 0x024F),
        'cyrillic': UnicodeBlock('Cyrillic', 0x0400, 0x04FF),
        'greek': UnicodeBlock('Greek', 0x0370, 0x03FF),
        'arabic': UnicodeBlock('Arabic', 0x0600, 0x06FF),
        'hebrew': UnicodeBlock('Hebrew', 0x0590, 0x05FF),
        'devanagari': UnicodeBlock('Devanagari', 0x0900, 0x097F),
        'bengali': UnicodeBlock('Bengali', 0x0980, 0x09FF),
        'thai': UnicodeBlock('Thai', 0x0E00, 0x0E7F),
        'ethiopic': UnicodeBlock('Ethiopic', 0x1200, 0x137F),
        'cherokee': UnicodeBlock('Cherokee', 0x13A0, 0x13FF),
        'hiragana': UnicodeBlock('Hiragana', 0x3040, 0x309F),
        'katakana': UnicodeBlock('Katakana', 0x30A0, 0x30FF),
        'cjk_unified': UnicodeBlock('CJK Unified', 0x4E00, 0x9FFF),
        'hangul': UnicodeBlock('Hangul Syllables', 0xAC00, 0xD7AF),
        'symbols_math': UnicodeBlock('Mathematical Symbols', 0x2200, 0x22FF),
        'symbols_arrows': UnicodeBlock('Arrows', 0x2190, 0x21FF),
        'symbols_misc': UnicodeBlock('Miscellaneous Symbols', 0x2600, 0x26FF),
        'symbols_geometric': UnicodeBlock('Geometric Shapes', 0x25A0, 0x25FF),
        'emoji_basic': UnicodeBlock('Emoji Basic', 0x1F300, 0x1F5FF),
        'emoji_people': UnicodeBlock('Emoji People', 0x1F600, 0x1F64F),
    }

    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 12,
        use_blocks: List[str] = None,
        seed: int = None
    ):
        """
        Initialize the word generator.

        Args:
            min_length: Minimum word length in characters
            max_length: Maximum word length in characters
            use_blocks: List of Unicode block names to use (None = all blocks)
            seed: Random seed for reproducibility (None = random)
        """
        self.min_length = min_length
        self.max_length = max_length

        if seed is not None:
            random.seed(seed)

        # Select Unicode blocks to use
        if use_blocks is None:
            self.active_blocks = list(self.UNICODE_BLOCKS.values())
        else:
            self.active_blocks = [
                self.UNICODE_BLOCKS[block]
                for block in use_blocks
                if block in self.UNICODE_BLOCKS
            ]

        if not self.active_blocks:
            raise ValueError("No valid Unicode blocks selected")

    def generate_word(self, length: int = None) -> str:
        """
        Generate a single random Unicode word.

        Args:
            length: Specific length (None = random between min and max)

        Returns:
            A random Unicode string
        """
        if length is None:
            length = random.randint(self.min_length, self.max_length)

        word = ''
        for _ in range(length):
            block = random.choice(self.active_blocks)
            try:
                word += block.generate_char()
            except ValueError:
                # Skip invalid Unicode ranges
                continue

        return word if word else self.generate_word(length)

    def generate_words(self, count: int, ensure_unique: bool = True) -> List[str]:
        """
        Generate multiple random words.

        Args:
            count: Number of words to generate
            ensure_unique: Ensure all words are unique

        Returns:
            List of random Unicode strings
        """
        if ensure_unique:
            words: Set[str] = set()
            attempts = 0
            max_attempts = count * 100  # Prevent infinite loops

            while len(words) < count and attempts < max_attempts:
                words.add(self.generate_word())
                attempts += 1

            if len(words) < count:
                raise RuntimeError(
                    f"Could not generate {count} unique words. "
                    f"Only generated {len(words)}. Try increasing word length range."
                )

            return list(words)
        else:
            return [self.generate_word() for _ in range(count)]

    def generate_word_pool(
        self,
        concept_id: str,
        pool_size: int = 5000
    ) -> List[str]:
        """
        Generate a pool of unique words for a single concept.

        This creates the "one-time pad" for a concept - thousands of
        unique random words that all map to the same semantic meaning.

        Args:
            concept_id: The concept this pool represents
            pool_size: Number of unique words to generate

        Returns:
            List of unique random words
        """
        return self.generate_words(pool_size, ensure_unique=True)

    def get_available_blocks(self) -> List[str]:
        """Get names of all available Unicode blocks."""
        return list(self.UNICODE_BLOCKS.keys())

    def __repr__(self):
        return (
            f"WordGenerator(length={self.min_length}-{self.max_length}, "
            f"blocks={len(self.active_blocks)})"
        )
