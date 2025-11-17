"""
Test script for GREMLIN core components.
Verifies concept dictionary, word generator, and language pack creation.
"""

from pathlib import Path
from core import ConceptDictionary, WordGenerator, LanguagePack


def test_concept_dictionary():
    """Test the concept dictionary."""
    print("\n=== Testing Concept Dictionary ===")

    cd = ConceptDictionary()
    print(f"Loaded: {cd}")
    print(f"Categories: {', '.join(cd.get_categories())}")

    # Test searching
    auth_concepts = cd.search_by_term("checking")
    print(f"\nSearch 'checking': {len(auth_concepts)} results")
    for concept in auth_concepts[:3]:
        print(f"  - {concept.id}: {concept.term}")

    # Test category retrieval
    actions = cd.get_concepts_by_category('actions')
    print(f"\nActions category: {len(actions)} concepts")

    return cd


def test_word_generator():
    """Test the word generator."""
    print("\n=== Testing Word Generator ===")

    # Test with limited Unicode blocks for demo
    wg = WordGenerator(
        min_length=4,
        max_length=10,
        use_blocks=['latin_basic', 'cyrillic', 'greek', 'hiragana']
    )
    print(f"Generator: {wg}")

    # Generate some sample words
    print("\nSample words:")
    for i in range(10):
        word = wg.generate_word()
        print(f"  {i+1}. {word}")

    # Test uniqueness
    print("\nTesting uniqueness (generating 1000 words)...")
    words = wg.generate_words(1000, ensure_unique=True)
    print(f"  Generated {len(words)} unique words")
    print(f"  Sample: {', '.join(words[:5])}")

    return wg


def test_language_pack(cd: ConceptDictionary, wg: WordGenerator):
    """Test language pack generation."""
    print("\n=== Testing Language Pack Generation ===")

    # Generate a small pack for testing (fewer words per concept)
    print("\nGenerating small test pack (100 words per concept)...")
    pack = LanguagePack.generate(
        concept_dict=cd,
        words_per_concept=100,  # Small for testing
        word_generator=wg
    )

    print(f"\n{pack}")

    # Test stats
    stats = pack.get_stats()
    print(f"\nLanguage Pack Stats:")
    print(f"  Language ID: {stats['language_id']}")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Usage: {stats['usage_percentage']:.2f}%")
    print(f"  Words remaining: {stats['words_remaining']:,}")

    # Test word retrieval
    print("\n=== Testing Word Retrieval ===")
    test_concepts = ['act_001', 'id_001', 'name_001', 'comp_001']

    for concept_id in test_concepts:
        concept = cd.get_concept(concept_id)
        word = pack.get_word_for_concept(concept_id)
        print(f"  {concept.term:20} -> {word}")

    # Save the pack
    output_path = Path('data/test_language_pack.json')
    pack.save(output_path)

    # Test loading
    print("\n=== Testing Pack Load ===")
    loaded_pack = LanguagePack.load(output_path)
    print(f"Loaded: {loaded_pack}")

    loaded_stats = loaded_pack.get_stats()
    print(f"Usage after load: {loaded_stats['usage_percentage']:.2f}%")

    return pack


def main():
    """Run all tests."""
    print("=" * 60)
    print("GREMLIN Core Component Tests")
    print("=" * 60)

    cd = test_concept_dictionary()
    wg = test_word_generator()
    pack = test_language_pack(cd, wg)

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
