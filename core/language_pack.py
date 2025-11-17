"""
Language Pack Management
Creates, saves, and loads synthetic language packs.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict, field
from enum import Enum

from .concepts import ConceptDictionary, Concept
from .word_generator import WordGenerator


class WordStatus(str, Enum):
    """Status of a word in the usage pool."""
    UNUSED = "unused"
    USED = "used"
    USE_LAST = "use_last"  # Words from potential DDoS attempts


@dataclass
class WordPool:
    """Pool of words for a single concept with usage tracking."""
    concept_id: str
    words: Dict[str, WordStatus]  # word -> status

    def get_unused_word(self) -> Optional[str]:
        """Get a random unused word from the pool."""
        import random
        unused = [w for w, status in self.words.items() if status == WordStatus.UNUSED]
        if unused:
            return random.choice(unused)

        # If no unused words, try use_last pool
        use_last = [w for w, status in self.words.items() if status == WordStatus.USE_LAST]
        if use_last:
            return random.choice(use_last)

        return None

    def mark_used(self, word: str):
        """Mark a word as used (burned)."""
        if word in self.words:
            self.words[word] = WordStatus.USED

    def mark_use_last(self, word: str):
        """Mark a word for use-last pool (potential attack)."""
        if word in self.words and self.words[word] == WordStatus.UNUSED:
            self.words[word] = WordStatus.USE_LAST

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics for this pool."""
        stats = {status.value: 0 for status in WordStatus}
        for status in self.words.values():
            stats[status.value] += 1
        return stats

    def words_remaining(self) -> int:
        """Count of unused + use_last words."""
        return sum(1 for s in self.words.values() if s in [WordStatus.UNUSED, WordStatus.USE_LAST])

    def usage_percentage(self) -> float:
        """Percentage of words used."""
        total = len(self.words)
        used = sum(1 for s in self.words.values() if s == WordStatus.USED)
        return (used / total * 100) if total > 0 else 0


@dataclass
class GrammarRules:
    """Grammar rules for a synthetic language."""
    word_order: str = "SVO"  # Subject-Verb-Object, SOV, VSO, etc.
    use_articles: bool = True
    use_prepositions: bool = True
    adjective_position: str = "before"  # "before" or "after" noun

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class LanguagePack:
    """
    A complete synthetic language pack.

    Contains all word pools, grammar rules, and metadata needed for
    two AI models to communicate in this ephemeral language.
    """
    language_id: str
    created_at: str
    grammar: GrammarRules
    word_pools: Dict[str, WordPool]  # concept_id -> WordPool
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def generate(
        cls,
        concept_dict: ConceptDictionary,
        words_per_concept: int = 5000,
        grammar_rules: Optional[GrammarRules] = None,
        word_generator: Optional[WordGenerator] = None,
        language_id: Optional[str] = None
    ) -> 'LanguagePack':
        """
        Generate a new language pack.

        Args:
            concept_dict: The concept dictionary to use
            words_per_concept: Number of words to generate per concept
            grammar_rules: Grammar rules (None = default SVO)
            word_generator: Word generator instance (None = create default)
            language_id: Specific language ID (None = generate UUID)

        Returns:
            A new LanguagePack
        """
        if language_id is None:
            language_id = str(uuid.uuid4())

        if grammar_rules is None:
            grammar_rules = GrammarRules()

        if word_generator is None:
            word_generator = WordGenerator()

        # Generate word pools for each concept
        word_pools = {}
        total_concepts = concept_dict.total_concepts()

        print(f"Generating language pack: {language_id}")
        print(f"Concepts: {total_concepts}")
        print(f"Words per concept: {words_per_concept}")
        print(f"Total words: {total_concepts * words_per_concept:,}")

        for i, concept in enumerate(concept_dict.get_all_concepts(), 1):
            # Generate unique words for this concept
            words = word_generator.generate_word_pool(
                concept.id,
                pool_size=words_per_concept
            )

            # Create word pool with all words marked as unused
            word_pools[concept.id] = WordPool(
                concept_id=concept.id,
                words={word: WordStatus.UNUSED for word in words}
            )

            if i % 50 == 0 or i == total_concepts:
                print(f"  Progress: {i}/{total_concepts} concepts")

        metadata = {
            'total_concepts': total_concepts,
            'words_per_concept': words_per_concept,
            'total_words': total_concepts * words_per_concept,
            'generator': str(word_generator)
        }

        return cls(
            language_id=language_id,
            created_at=datetime.utcnow().isoformat(),
            grammar=grammar_rules,
            word_pools=word_pools,
            metadata=metadata
        )

    def get_word_for_concept(self, concept_id: str) -> Optional[str]:
        """
        Get an unused word for a concept and mark it as used.

        This is the core translation function - maps concept to random word.
        """
        pool = self.word_pools.get(concept_id)
        if pool is None:
            return None

        word = pool.get_unused_word()
        if word:
            pool.mark_used(word)

        return word

    def get_stats(self) -> Dict:
        """Get overall language pack statistics."""
        total_words = sum(len(pool.words) for pool in self.word_pools.values())
        total_used = sum(
            sum(1 for s in pool.words.values() if s == WordStatus.USED)
            for pool in self.word_pools.values()
        )

        return {
            'language_id': self.language_id,
            'total_concepts': len(self.word_pools),
            'total_words': total_words,
            'total_used': total_used,
            'usage_percentage': (total_used / total_words * 100) if total_words > 0 else 0,
            'words_remaining': total_words - total_used
        }

    def save(self, path: Path, encrypt: bool = False):
        """
        Save language pack to file.

        Args:
            path: Output file path
            encrypt: Whether to encrypt the pack (TODO: implement encryption)
        """
        # Convert to serializable format
        data = {
            'language_id': self.language_id,
            'created_at': self.created_at,
            'grammar': self.grammar.to_dict(),
            'word_pools': {
                concept_id: {
                    'concept_id': pool.concept_id,
                    'words': {word: status.value for word, status in pool.words.items()}
                }
                for concept_id, pool in self.word_pools.items()
            },
            'metadata': self.metadata
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Language pack saved to: {path}")
        print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")

    @classmethod
    def load(cls, path: Path) -> 'LanguagePack':
        """
        Load language pack from file.

        Args:
            path: Input file path

        Returns:
            Loaded LanguagePack
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct word pools
        word_pools = {}
        for concept_id, pool_data in data['word_pools'].items():
            word_pools[concept_id] = WordPool(
                concept_id=pool_data['concept_id'],
                words={
                    word: WordStatus(status)
                    for word, status in pool_data['words'].items()
                }
            )

        return cls(
            language_id=data['language_id'],
            created_at=data['created_at'],
            grammar=GrammarRules.from_dict(data['grammar']),
            word_pools=word_pools,
            metadata=data.get('metadata', {})
        )

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"LanguagePack(id={self.language_id[:8]}..., "
            f"concepts={stats['total_concepts']}, "
            f"usage={stats['usage_percentage']:.1f}%)"
        )
