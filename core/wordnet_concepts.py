"""
WordNet Synset Extractor
Converts WordNet synsets into GREMLIN concepts.
"""

from typing import List, Set
from dataclasses import dataclass
from nltk.corpus import wordnet as wn
import random


@dataclass
class Concept:
    """Represents a semantic concept (compatible with ConceptDictionary)."""
    id: str
    category: str
    description: str
    examples: List[str] = None


class WordNetConceptExtractor:
    """
    Extracts concepts from WordNet synsets.
    Ranks by frequency/usefulness to select top N synsets.
    """

    # POS categories
    POS_NAMES = {
        'n': 'noun',
        'v': 'verb',
        'a': 'adjective',
        's': 'adjective_satellite',
        'r': 'adverb'
    }

    def __init__(self):
        """Initialize the extractor."""
        self.all_synsets = list(wn.all_synsets())
        print(f"Loaded {len(self.all_synsets)} synsets from WordNet")

    def get_synset_frequency_score(self, synset) -> float:
        """
        Calculate a frequency score for a synset.

        Heuristics:
        - Nouns and verbs are more common than adjectives/adverbs
        - Higher depth = more specific (lower score)
        - More lemmas = more common (higher score)
        - Shorter name = more basic/common (higher score)
        """
        score = 100.0

        # POS weighting (nouns and verbs most common)
        pos_weights = {
            'n': 1.0,   # Nouns
            'v': 0.9,   # Verbs
            'a': 0.5,   # Adjectives
            's': 0.5,   # Adjective satellites
            'r': 0.3    # Adverbs (least common)
        }
        score *= pos_weights.get(synset.pos(), 0.1)

        # Depth penalty (shallower = more general = more common)
        try:
            min_depth = synset.min_depth()
            if min_depth:
                score *= (1.0 / (1.0 + min_depth * 0.1))
        except:
            pass

        # Lemma count bonus (more words = more common concept)
        lemma_count = len(synset.lemmas())
        score *= (1.0 + lemma_count * 0.2)

        # Name length penalty (shorter = more basic)
        name_parts = synset.name().split('.')[0]
        score *= (1.0 / (1.0 + len(name_parts) * 0.01))

        return score

    def extract_top_synsets(self, n: int, progress_callback=None) -> List[Concept]:
        """
        Extract top N most useful synsets as Concepts.

        Args:
            n: Number of synsets to extract
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of Concept objects
        """
        print(f"\nRanking {len(self.all_synsets)} synsets by frequency...")

        # Score all synsets
        scored_synsets = []
        for i, synset in enumerate(self.all_synsets):
            if progress_callback and i % 5000 == 0:
                progress_callback(i, len(self.all_synsets), f"Scoring synsets: {i}/{len(self.all_synsets)}")

            score = self.get_synset_frequency_score(synset)
            scored_synsets.append((score, synset))

        # Sort by score (descending)
        scored_synsets.sort(key=lambda x: x[0], reverse=True)

        # Take top N
        print(f"Selecting top {n} synsets...")
        top_synsets = [synset for score, synset in scored_synsets[:n]]

        # Convert to Concept objects
        concepts = []
        for i, synset in enumerate(top_synsets):
            if progress_callback and i % 100 == 0:
                progress_callback(i, n, f"Converting synsets: {i}/{n}")

            # Generate concept ID
            # Format: wordnet_{word}_{pos}_{num}
            concept_id = f"wordnet_{synset.name().replace('.', '_')}"

            # Get category from POS
            category = f"wordnet_{self.POS_NAMES.get(synset.pos(), 'unknown')}"

            # Get definition
            description = synset.definition()

            # Get example sentences
            examples = synset.examples()[:3]  # Limit to 3 examples

            concept = Concept(
                id=concept_id,
                category=category,
                description=description,
                examples=examples if examples else []
            )
            concepts.append(concept)

        print(f"✅ Extracted {len(concepts)} concepts from WordNet")
        return concepts

    def get_all_concepts(self, progress_callback=None) -> List[Concept]:
        """
        Get ALL WordNet synsets as concepts (117K+).
        This will take a while!

        Args:
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of all Concept objects
        """
        print(f"\n⚠️  Extracting ALL {len(self.all_synsets)} WordNet synsets...")
        print("This may take several minutes...")

        concepts = []
        for i, synset in enumerate(self.all_synsets):
            if progress_callback and i % 1000 == 0:
                progress_callback(i, len(self.all_synsets), f"Converting synsets: {i}/{len(self.all_synsets)}")

            # Generate concept ID
            concept_id = f"wordnet_{synset.name().replace('.', '_')}"

            # Get category from POS
            category = f"wordnet_{self.POS_NAMES.get(synset.pos(), 'unknown')}"

            # Get definition
            description = synset.definition()

            # Get example sentences
            examples = synset.examples()[:3]

            concept = Concept(
                id=concept_id,
                category=category,
                description=description,
                examples=examples if examples else []
            )
            concepts.append(concept)

        print(f"✅ Extracted {len(concepts)} concepts from WordNet")
        return concepts

    def get_stats(self) -> dict:
        """Get statistics about WordNet corpus."""
        return {
            'total_synsets': len(self.all_synsets),
            'nouns': len(list(wn.all_synsets('n'))),
            'verbs': len(list(wn.all_synsets('v'))),
            'adjectives': len(list(wn.all_synsets('a'))),
            'adverbs': len(list(wn.all_synsets('r')))
        }


def test_extractor():
    """Test the WordNet concept extractor."""
    extractor = WordNetConceptExtractor()

    stats = extractor.get_stats()
    print("\nWordNet Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    # Test with small extraction
    print("\n" + "="*60)
    print("Testing: Extract top 10 synsets")
    print("="*60)

    concepts = extractor.extract_top_synsets(10)

    for concept in concepts:
        print(f"\nID: {concept.id}")
        print(f"Category: {concept.category}")
        print(f"Description: {concept.description}")
        if concept.examples:
            print(f"Examples: {', '.join(concept.examples[:2])}")


if __name__ == '__main__':
    test_extractor()
