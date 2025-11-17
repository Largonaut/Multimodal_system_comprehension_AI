#!/usr/bin/env python3
"""
Production Language Pack Generator
Generates full-scale GREMLIN language packs for deployment.
"""

import argparse
from pathlib import Path
from datetime import datetime
from core import ConceptDictionary, WordGenerator, LanguagePack, GrammarEngine


def generate_pack(
    output_dir: Path,
    words_per_concept: int = 5000,
    language_id: str = None,
    word_order: str = "SVO",
    unicode_blocks: list = None
):
    """
    Generate a production language pack.

    Args:
        output_dir: Directory to save the pack
        words_per_concept: Number of words per concept (default 5000)
        language_id: Specific language ID (default: auto-generate)
        word_order: Grammar word order (SVO, SOV, VSO, etc.)
        unicode_blocks: List of Unicode blocks to use (None = all)
    """
    print("=" * 70)
    print("GREMLIN Production Language Pack Generator")
    print("=" * 70)

    # Load concept dictionary
    print("\n[1/4] Loading concept dictionary...")
    cd = ConceptDictionary()
    print(f"      Loaded {cd.total_concepts()} concepts")

    # Create word generator
    print("\n[2/4] Initializing word generator...")
    if unicode_blocks is None:
        # Default: use diverse Unicode blocks
        unicode_blocks = [
            'latin_basic', 'latin_extended_a', 'cyrillic', 'greek',
            'arabic', 'hebrew', 'hiragana', 'katakana',
            'symbols_math', 'symbols_misc'
        ]

    wg = WordGenerator(
        min_length=4,
        max_length=15,
        use_blocks=unicode_blocks
    )
    print(f"      {wg}")
    print(f"      Unicode blocks: {', '.join(unicode_blocks)}")

    # Generate language pack
    print(f"\n[3/4] Generating language pack...")
    print(f"      Words per concept: {words_per_concept:,}")
    print(f"      Total words: {cd.total_concepts() * words_per_concept:,}")
    print(f"      Grammar: {word_order}")
    print()

    from core.language_pack import GrammarRules
    grammar = GrammarRules(word_order=word_order)

    pack = LanguagePack.generate(
        concept_dict=cd,
        words_per_concept=words_per_concept,
        grammar_rules=grammar,
        word_generator=wg,
        language_id=language_id
    )

    # Save pack
    print(f"\n[4/4] Saving language pack...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include metadata in filename for easy identification
    filename = f"language_pack_{words_per_concept}w_{word_order}_{timestamp}.json"
    output_path = output_dir / filename

    pack.save(output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("Language Pack Generated Successfully!")
    print("=" * 70)

    stats = pack.get_stats()
    print(f"\nLanguage ID: {stats['language_id']}")
    print(f"Output File: {output_path}")
    print(f"File Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\nCapacity:")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Words per concept: {words_per_concept:,}")

    # Estimate authentication rounds
    # Approximate: ~6-8 words per authentication exchange (both directions)
    words_per_exchange = 8
    estimated_rounds = stats['total_words'] // words_per_exchange
    print(f"\n  Estimated authentication rounds: ~{estimated_rounds:,}")

    print("\n" + "=" * 70)

    return pack


def main():
    parser = argparse.ArgumentParser(
        description="Generate GREMLIN synthetic language packs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default pack (5000 words/concept)
  python generate_language_pack.py

  # Generate large pack (10000 words/concept)
  python generate_language_pack.py --words 10000

  # Generate pack with specific grammar
  python generate_language_pack.py --grammar SOV

  # Generate lightweight pack for testing
  python generate_language_pack.py --words 500 --output test_packs/
        """
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('language_packs'),
        help='Output directory (default: language_packs/)'
    )

    parser.add_argument(
        '--words', '-w',
        type=int,
        default=5000,
        help='Words per concept (default: 5000)'
    )

    parser.add_argument(
        '--grammar', '-g',
        type=str,
        default='SVO',
        choices=['SVO', 'SOV', 'VSO', 'VOS', 'OVS', 'OSV'],
        help='Word order grammar (default: SVO)'
    )

    parser.add_argument(
        '--id',
        type=str,
        default=None,
        help='Specific language ID (default: auto-generate UUID)'
    )

    args = parser.parse_args()

    generate_pack(
        output_dir=args.output,
        words_per_concept=args.words,
        language_id=args.id,
        word_order=args.grammar
    )


if __name__ == '__main__':
    main()
