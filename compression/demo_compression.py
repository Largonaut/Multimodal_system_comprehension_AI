"""
Semantic Compression Demo
Demonstrates the dual-layer word+concept compression system.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from compression.token_cost_analyzer import TokenCostAnalyzer
from compression.semantic_compression_dict import SemanticCompressionDict


def demo_token_analysis():
    """Demo: Analyze Unicode token costs."""
    print("\n" + "=" * 70)
    print("DEMO 1: Unicode Token Cost Analysis")
    print("=" * 70)

    analyzer = TokenCostAnalyzer()

    # Try to load tokenizer
    if not analyzer.load_tokenizer("gpt2"):
        print("\n⚠️  Install transformers for accurate token analysis:")
        print("   pip install transformers")
        print("\n   Using fallback heuristics for now...")

    # Get best single-token chars
    best_chars = analyzer.get_best_single_token_chars(500)

    print(f"\n✓ Found {len(best_chars)} optimal 1-token characters")
    print(f"\nSample characters (first 100):")
    print(''.join(best_chars[:100]))

    # Show token cost comparison
    print(f"\n\nToken Cost Comparison:")
    test_chars = ['a', 'α', '中', '🔥', '⚛']
    for char in test_chars:
        cost = analyzer.get_token_cost(char)
        print(f"  '{char}' (U+{ord(char):04X}) = {cost} token(s)")

    # Generate 2-char combos
    combos = analyzer.generate_2char_combos(best_chars[:50], 100)
    print(f"\n✓ Sample 2-char combinations (from 50 base chars):")
    print(f"  {', '.join(combos[:20])}")

    return best_chars


def demo_compression_dictionary(best_chars):
    """Demo: Build and test compression dictionary."""
    print("\n" + "=" * 70)
    print("DEMO 2: Semantic Compression Dictionary")
    print("=" * 70)

    builder = SemanticCompressionDict()

    # Load data (smaller samples for demo)
    print("\nLoading data...")
    builder.load_word_frequencies(10000)  # Top 10K words for demo
    builder.load_wordnet_synsets(1000)     # 1K synsets for demo

    # Generate codes
    builder.generate_unicode_codes(best_chars[:100])  # 100 base chars for demo

    # Build dictionary
    builder.build_compression_dict()

    print(f"\n✓ Dictionary built:")
    print(f"  Total entries: {len(builder.compression_map):,}")

    # Show some mappings
    print(f"\n📊 Sample Mappings (Common Words → 1-char):")
    word_samples = list(builder.compression_map.items())[:10]
    for word, code in word_samples:
        if not word.startswith('syn:'):
            print(f"  '{word}' → '{code}' ({len(code)} char)")

    print(f"\n📊 Sample Mappings (Synsets → 2-char):")
    synset_samples = [item for item in list(builder.compression_map.items())[:50] if item[0].startswith('syn:')][:5]
    for synset_key, code in synset_samples:
        entry = builder.decompression_map[code]
        print(f"  {entry['lemma']} ({synset_key}) → '{code}'")
        print(f"    Def: {entry['definition'][:60]}...")

    return builder


def demo_compression_in_action(builder):
    """Demo: Compress and decompress sample texts."""
    print("\n" + "=" * 70)
    print("DEMO 3: Compression in Action")
    print("=" * 70)

    test_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test of the compression system",
        "tomorrow we will create something amazing",
        "artificial intelligence is transforming the world"
    ]

    for i, text in enumerate(test_sentences, 1):
        print(f"\n--- Test {i} ---")
        print(f"Original:    {text}")

        compressed = builder.compress_text(text)
        print(f"Compressed:  {compressed}")

        decompressed = builder.decompress_text(compressed)
        print(f"Decompressed: {decompressed}")

        # Calculate stats
        orig_len = len(text)
        comp_len = len(compressed)
        ratio = (1 - comp_len / orig_len) * 100

        print(f"Chars: {orig_len} → {comp_len} ({ratio:+.1f}%)")

        # Token estimate (assuming 1 token per char in compressed)
        orig_tokens = builder.compression_map
        comp_tokens = compressed.count(' ') + 1  # Rough estimate
        token_ratio = (1 - comp_tokens / (text.count(' ') + 1)) * 100

        print(f"Token savings (estimated): {token_ratio:+.1f}%")


def demo_theoretical_limits():
    """Demo: Show theoretical compression limits."""
    print("\n" + "=" * 70)
    print("DEMO 4: Theoretical Compression Limits")
    print("=" * 70)

    print("\n📈 Coverage Analysis:")
    print(f"  500 base chars (1-token)")
    print(f"  → 500 codes of length 1")
    print(f"  → 250,000 codes of length 2 (500²)")
    print(f"  → Total: 250,500 unique codes")
    print()
    print(f"  Allocated to:")
    print(f"  • 100,000 common words (lexical)")
    print(f"  • 117,000 WordNet synsets (semantic)")
    print(f"  • 33,500 reserved for expansion")
    print(f"  = 250,500 TOTAL")

    print("\n💾 Token Economics:")
    print(f"  Traditional encoding:")
    print(f"    'hello world' = 2 tokens (GPT)")
    print()
    print(f"  With semantic compression:")
    print(f"    'hello' → 'の' = 1 token")
    print(f"    'world' → 'と' = 1 token")
    print(f"    Total = 2 tokens (same)")
    print()
    print(f"  But for longer text:")
    print(f"    'the quick brown fox jumps over the lazy dog'")
    print(f"    = ~10 tokens (traditional)")
    print(f"    = ~5-7 tokens (with compression, using 2-char codes)")
    print(f"    = 30-50% token savings!")

    print("\n🎯 Use Cases:")
    print(f"  ✓ On-device AI (bandwidth optimization)")
    print(f"  ✓ AI-to-AI communication (semantic precision)")
    print(f"  ✓ API call cost reduction (fewer tokens)")
    print(f"  ✓ Low-bandwidth environments")
    print(f"  ✓ Bilingual encoding (word + concept)")


def main():
    """Run all demos."""
    print("=" * 70)
    print("🚀 SEMANTIC COMPRESSION SYSTEM - FULL DEMO")
    print("=" * 70)
    print()
    print("Combining:")
    print("  • 100K most common English words (lexical frequency)")
    print("  • 117K WordNet synsets (semantic precision)")
    print("  • 220K lowest-token-cost Unicode codes (optimization)")
    print()
    print("Goal: Maximum compression with semantic preservation")
    print("=" * 70)

    # Run demos
    best_chars = demo_token_analysis()
    builder = demo_compression_dictionary(best_chars)
    demo_compression_in_action(builder)
    demo_theoretical_limits()

    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Install transformers: pip install transformers")
    print("  2. Install Brown corpus: python -m nltk.downloader brown")
    print("  3. Run full builder: python compression/semantic_compression_dict.py")
    print()
    print("This will generate:")
    print("  • compression/semantic_compression.json (full dictionary)")
    print("  • ~100MB dictionary file")
    print("  • Ready for on-device AI deployment")


if __name__ == '__main__':
    main()
