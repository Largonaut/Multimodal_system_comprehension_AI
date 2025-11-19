"""
Semantic Compression Dictionary Builder
Creates optimal word+concept→Unicode mapping for AI-to-AI compression.

Combines:
- 100K most common English words (lexical frequency)
- 117K WordNet synsets (semantic precision)
"""

from typing import Dict, List, Tuple
from pathlib import Path
from collections import Counter
import json


class SemanticCompressionDict:
    """Build compression dictionary optimized for token cost and semantic coverage."""

    def __init__(self):
        """Initialize the builder."""
        self.word_freq = {}
        self.synsets = []
        self.compression_map = {}  # English → Unicode
        self.decompression_map = {}  # Unicode → English
        self.unicode_codes = []

    def load_word_frequencies(self, top_n: int = 100000):
        """
        Load most common English words from Brown corpus.

        Args:
            top_n: Number of top words to include
        """
        try:
            from nltk.corpus import brown
            print(f"Loading top {top_n:,} words from Brown corpus...")

            words = [word.lower() for word in brown.words()]
            freq_dist = Counter(words)

            # Get top N most common words
            self.word_freq = dict(freq_dist.most_common(top_n))

            print(f"✓ Loaded {len(self.word_freq):,} common words")
            print(f"  Most common: {', '.join(list(self.word_freq.keys())[:10])}")

        except ImportError:
            print("⚠️  Brown corpus not available")
            print("   Run: python -m nltk.downloader brown")

    def load_wordnet_synsets(self, top_n: int = 117659):
        """
        Load WordNet synsets.

        Args:
            top_n: Number of synsets to include (default: all)
        """
        try:
            from core.wordnet_concepts import WordNetConceptExtractor

            print(f"Loading WordNet synsets...")
            extractor = WordNetConceptExtractor()

            if top_n >= len(extractor.all_synsets):
                # Get all synsets
                self.synsets = extractor.all_synsets
            else:
                # Get top N by frequency
                concepts = extractor.extract_top_synsets(top_n)
                # Convert back to synsets
                self.synsets = [
                    extractor.wn.synset(c.id.replace('wordnet_', '').replace('_', '.', 2))
                    for c in concepts
                ]

            print(f"✓ Loaded {len(self.synsets):,} WordNet synsets")

        except Exception as e:
            print(f"⚠️  Could not load WordNet: {e}")

    def generate_unicode_codes(self, single_token_chars: List[str]):
        """
        Generate Unicode codes using 1-char and 2-char combinations.

        Args:
            single_token_chars: List of optimal 1-token Unicode characters
        """
        print(f"Generating Unicode codes from {len(single_token_chars)} base chars...")

        # 1-char codes (most common items get shortest codes)
        self.unicode_codes = single_token_chars.copy()

        # 2-char codes
        for c1 in single_token_chars:
            for c2 in single_token_chars:
                self.unicode_codes.append(c1 + c2)
                if len(self.unicode_codes) >= 220000:
                    break
            if len(self.unicode_codes) >= 220000:
                break

        print(f"✓ Generated {len(self.unicode_codes):,} Unicode codes")
        print(f"  1-char: {len(single_token_chars):,}")
        print(f"  2-char: {len(self.unicode_codes) - len(single_token_chars):,}")

    def build_compression_dict(self):
        """
        Build the compression dictionary.

        Strategy:
        1. Most common words get 1-char codes
        2. Less common words get 2-char codes
        3. WordNet synsets get 2-char codes (for semantic precision)
        """
        print("\nBuilding compression dictionary...")

        code_idx = 0
        total_entries = len(self.word_freq) + len(self.synsets)

        # Layer 1: Common words (sorted by frequency)
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        for word, freq in sorted_words:
            if code_idx >= len(self.unicode_codes):
                break

            code = self.unicode_codes[code_idx]
            self.compression_map[word] = code
            self.decompression_map[code] = {
                'type': 'word',
                'value': word,
                'freq': freq
            }
            code_idx += 1

        print(f"✓ Mapped {len(sorted_words):,} common words")

        # Layer 2: WordNet synsets
        for synset in self.synsets:
            if code_idx >= len(self.unicode_codes):
                break

            # Use synset name as key
            synset_key = f"syn:{synset.name()}"
            code = self.unicode_codes[code_idx]

            # Get primary lemma
            primary_lemma = synset.lemmas()[0].name()

            self.compression_map[synset_key] = code
            self.decompression_map[code] = {
                'type': 'synset',
                'value': synset.name(),
                'lemma': primary_lemma,
                'definition': synset.definition()
            }
            code_idx += 1

        print(f"✓ Mapped {len(self.synsets):,} WordNet synsets")
        print(f"\nTotal dictionary size: {len(self.compression_map):,} entries")

    def compress_text(self, text: str, use_semantic: bool = False) -> str:
        """
        Compress English text to Unicode.

        Args:
            text: English text to compress
            use_semantic: Use semantic (synset) compression when available

        Returns:
            Compressed Unicode string
        """
        words = text.lower().split()
        compressed = []

        for word in words:
            # Try direct word lookup
            if word in self.compression_map:
                compressed.append(self.compression_map[word])
            else:
                # Word not in dictionary, keep as-is
                compressed.append(word)

        return ' '.join(compressed)

    def decompress_text(self, compressed: str) -> str:
        """
        Decompress Unicode back to English.

        Args:
            compressed: Compressed Unicode string

        Returns:
            Decompressed English text
        """
        tokens = compressed.split()
        decompressed = []

        for token in tokens:
            if token in self.decompression_map:
                entry = self.decompression_map[token]
                if entry['type'] == 'word':
                    decompressed.append(entry['value'])
                elif entry['type'] == 'synset':
                    decompressed.append(entry['lemma'])
            else:
                # Unknown token, keep as-is
                decompressed.append(token)

        return ' '.join(decompressed)

    def analyze_compression_ratio(self, sample_texts: List[str]):
        """
        Analyze compression ratio on sample texts.

        Args:
            sample_texts: List of sample English texts to test
        """
        print("\n" + "=" * 60)
        print("Compression Analysis")
        print("=" * 60)

        total_original = 0
        total_compressed = 0

        for i, text in enumerate(sample_texts, 1):
            compressed = self.compress_text(text)
            decompressed = self.decompress_text(compressed)

            orig_chars = len(text)
            comp_chars = len(compressed)
            ratio = (1 - comp_chars / orig_chars) * 100 if orig_chars > 0 else 0

            total_original += orig_chars
            total_compressed += comp_chars

            print(f"\nSample {i}:")
            print(f"  Original:    {text[:80]}...")
            print(f"  Compressed:  {compressed[:80]}...")
            print(f"  Decompressed: {decompressed[:80]}...")
            print(f"  Chars: {orig_chars} → {comp_chars} ({ratio:+.1f}%)")

        overall_ratio = (1 - total_compressed / total_original) * 100
        print(f"\nOverall compression: {overall_ratio:+.1f}%")

    def save_dictionary(self, output_path: Path):
        """Save compression dictionary to file."""
        data = {
            'metadata': {
                'total_entries': len(self.compression_map),
                'word_count': sum(1 for k in self.compression_map.keys() if not k.startswith('syn:')),
                'synset_count': sum(1 for k in self.compression_map.keys() if k.startswith('syn:')),
            },
            'compression_map': self.compression_map,
            'decompression_map': self.decompression_map
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Saved dictionary to {output_path}")
        print(f"  File size: {file_size:.2f} MB")


def main():
    """Build semantic compression dictionary."""
    print("=" * 60)
    print("Semantic Compression Dictionary Builder")
    print("=" * 60)
    print()

    # Initialize builder
    builder = SemanticCompressionDict()

    # Load data
    builder.load_word_frequencies(100000)
    builder.load_wordnet_synsets(117659)

    # Generate Unicode codes
    from compression.token_cost_analyzer import TokenCostAnalyzer
    analyzer = TokenCostAnalyzer()
    analyzer.load_tokenizer("gpt2")
    best_chars = analyzer.get_best_single_token_chars(500)
    builder.generate_unicode_codes(best_chars)

    # Build dictionary
    builder.build_compression_dict()

    # Test compression
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "tomorrow we will create a temporal typing incident",
        "artificial intelligence and machine learning are transforming technology",
        "the future of on-device AI requires efficient compression"
    ]

    builder.analyze_compression_ratio(sample_texts)

    # Save dictionary
    output_path = Path(__file__).parent / "semantic_compression.json"
    builder.save_dictionary(output_path)


if __name__ == '__main__':
    main()
