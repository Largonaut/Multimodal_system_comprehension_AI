"""
Concept Dictionary Management
Loads and manages the permanent semantic foundation for GREMLIN.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Concept:
    """Represents a single semantic concept."""
    id: str
    term: str
    category: str
    variants: Optional[List[str]] = None

    def __post_init__(self):
        if self.variants is None:
            self.variants = [self.term]


class ConceptDictionary:
    """
    Manages the master concept dictionary.

    The concept dictionary is the permanent semantic layer that all
    synthetic languages map to. Each concept represents a meaning that
    can be expressed in thousands of random Unicode forms.
    """

    def __init__(self, concepts_file: Optional[Path] = None):
        """
        Initialize the concept dictionary.

        Args:
            concepts_file: Path to base_concepts.json. If None, uses default location.
        """
        if concepts_file is None:
            concepts_file = Path(__file__).parent.parent / 'data' / 'base_concepts.json'

        self.concepts_file = Path(concepts_file)
        self.concepts: Dict[str, Concept] = {}
        self.categories: Dict[str, List[str]] = {}

        self._load_concepts()

    def _load_concepts(self):
        """Load concepts from JSON file."""
        with open(self.concepts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse all concept categories
        for category_name, concept_list in data['concepts'].items():
            self.categories[category_name] = []

            for concept_data in concept_list:
                # Determine category from the concept_data or category_name
                cat = concept_data.get('category', category_name)

                concept = Concept(
                    id=concept_data['id'],
                    term=concept_data['term'],
                    category=cat,
                    variants=concept_data.get('variants', [concept_data['term']])
                )

                self.concepts[concept.id] = concept
                self.categories[category_name].append(concept.id)

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)

    def get_concepts_by_category(self, category: str) -> List[Concept]:
        """Get all concepts in a category."""
        concept_ids = self.categories.get(category, [])
        return [self.concepts[cid] for cid in concept_ids]

    def get_all_concepts(self) -> List[Concept]:
        """Get all concepts."""
        return list(self.concepts.values())

    def total_concepts(self) -> int:
        """Get total number of concepts."""
        return len(self.concepts)

    def get_categories(self) -> List[str]:
        """Get all category names."""
        return list(self.categories.keys())

    def search_by_term(self, term: str) -> List[Concept]:
        """Search for concepts by English term (fuzzy match)."""
        term_lower = term.lower()
        matches = []

        for concept in self.concepts.values():
            if term_lower in concept.term.lower():
                matches.append(concept)
            elif concept.variants:
                for variant in concept.variants:
                    if term_lower in variant.lower():
                        matches.append(concept)
                        break

        return matches

    def __repr__(self):
        return f"ConceptDictionary({self.total_concepts()} concepts, {len(self.categories)} categories)"
