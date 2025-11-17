"""
Test translation pipeline - English to Synthetic language.
"""

import random
from core import ConceptDictionary, WordGenerator, LanguagePack, GrammarEngine


def test_translation():
    """Test complete translation pipeline."""
    print("=" * 70)
    print("GREMLIN Translation Test")
    print("=" * 70)

    # Load components
    print("\n[1/5] Loading concept dictionary...")
    cd = ConceptDictionary()
    print(f"      Loaded {cd.total_concepts()} concepts")

    print("\n[2/5] Creating word generator...")
    wg = WordGenerator(
        min_length=4,
        max_length=12,
        use_blocks=['latin_basic', 'cyrillic', 'greek', 'arabic', 'hiragana', 'katakana']
    )
    print(f"      {wg}")

    print("\n[3/5] Generating language pack...")
    print("      (This may take a moment - generating 500 words per concept)")
    pack = LanguagePack.generate(
        concept_dict=cd,
        words_per_concept=500,  # Enough for demo
        word_generator=wg
    )
    stats = pack.get_stats()
    print(f"      Generated {stats['total_words']:,} words across {stats['total_concepts']} concepts")

    print("\n[4/5] Loading grammar engine...")
    grammar = GrammarEngine()
    print(f"      Grammar engine loaded")

    print("\n[5/5] Running authentication exchange demo...")
    print("=" * 70)

    # Run 10 authentication rounds
    num_rounds = 10
    print(f"\nGenerating {num_rounds} authentication exchanges:\n")

    # Get available IDs
    names = cd.get_concepts_by_category('variables_names')
    companies = cd.get_concepts_by_category('variables_companies')
    topics = cd.get_concepts_by_category('variables_topics')

    for i in range(1, num_rounds + 1):
        # Pick random variables
        name = random.choice(names)
        company = random.choice(companies)
        topic = random.choice(topics)

        # Generate authentication pair
        eng_client, syn_client, eng_server, syn_server = grammar.generate_authentication_pair(
            name.id,
            company.id,
            topic.id,
            pack,
            cd
        )

        print(f"Round {i}:")
        print(f"  CLIENT [EN]: {eng_client}")
        print(f"  CLIENT [SYN]: {syn_client}")
        print(f"  SERVER [EN]: {eng_server}")
        print(f"  SERVER [SYN]: {syn_server}")
        print()

    # Show final stats
    print("=" * 70)
    final_stats = pack.get_stats()
    print(f"\nLanguage Pack Usage:")
    print(f"  Total words: {final_stats['total_words']:,}")
    print(f"  Words used: {final_stats['total_used']:,}")
    print(f"  Usage: {final_stats['usage_percentage']:.2f}%")
    print(f"  Words remaining: {final_stats['words_remaining']:,}")

    # Calculate how many more rounds possible
    words_per_round = final_stats['total_used'] / num_rounds
    remaining_rounds = int(final_stats['words_remaining'] / words_per_round)
    print(f"\n  Estimated remaining authentication rounds: ~{remaining_rounds}")

    print("\n" + "=" * 70)
    print("Translation test complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_translation()
