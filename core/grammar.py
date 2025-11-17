"""
Grammar Engine
Applies linguistic rules to construct sentences in synthetic languages.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .language_pack import LanguagePack
from .concepts import ConceptDictionary


@dataclass
class SentenceComponent:
    """A component of a sentence."""
    type: str  # action, identity, connector, variable, punctuation
    concept: Optional[str] = None  # concept_id
    value: Optional[str] = None  # for punctuation or fixed values


class GrammarEngine:
    """
    Applies grammar rules to construct sentences in synthetic languages.

    Takes English sentence components and rearranges them according to
    the grammar rules of the synthetic language.
    """

    def __init__(self, grammar_file: Optional[Path] = None):
        """
        Initialize grammar engine.

        Args:
            grammar_file: Path to grammar_rules.json (None = default)
        """
        if grammar_file is None:
            grammar_file = Path(__file__).parent.parent / 'data' / 'grammar_rules.json'

        self.grammar_file = Path(grammar_file)
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict:
        """Load grammar rules from JSON."""
        with open(self.grammar_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_sentence_template(self, template_name: str) -> Dict[str, Any]:
        """Get a sentence template by name."""
        return self.rules['sentence_templates'].get(template_name)

    def translate_sentence(
        self,
        template_name: str,
        variables: Dict[str, str],
        language_pack: LanguagePack,
        concept_dict: ConceptDictionary
    ) -> str:
        """
        Translate a sentence from English to synthetic language.

        Args:
            template_name: Name of sentence template to use
            variables: Dictionary mapping variable names to concept_ids
                      (e.g., {'name': 'name_001', 'company': 'comp_001'})
            language_pack: The language pack to use for word lookup
            concept_dict: Concept dictionary for lookups

        Returns:
            Sentence in synthetic language
        """
        template = self.get_sentence_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Build word list according to template
        words = []
        for component_data in template['components']:
            component = SentenceComponent(**component_data)

            if component.type == 'punctuation':
                # Keep punctuation as-is (or could randomize this too)
                continue  # Skip punctuation for now

            # Determine concept_id
            if component.type == 'variable':
                # Use provided variable mapping
                concept_id = variables.get(component.concept)
            else:
                # Use concept from template
                concept_id = component.concept

            if not concept_id:
                continue

            # Get synthetic word from language pack
            word = language_pack.get_word_for_concept(concept_id)
            if word:
                words.append(word)

        # Apply word order transformation
        words = self._apply_word_order(words, language_pack.grammar.word_order)

        return ' '.join(words)

    def _apply_word_order(self, words: List[str], word_order: str) -> List[str]:
        """
        Apply word order transformation.

        For demo, we keep it simple - just reverse for SOV, shuffle for others.
        In a full implementation, we'd parse sentence structure and reorder.
        """
        if word_order == "SOV":
            # Reverse order (approximation of SOV)
            return list(reversed(words))
        elif word_order == "VSO":
            # Move first word to front (approximation of VSO)
            if len(words) > 1:
                return [words[-1]] + words[:-1]
            return words
        else:
            # SVO - keep original order
            return words

    def get_random_word_order(self) -> str:
        """Get a random word order pattern."""
        return random.choice(list(self.rules['word_orders'].keys()))

    def generate_authentication_pair(
        self,
        name_id: str,
        company_id: str,
        topic_id: str,
        language_pack: LanguagePack,
        concept_dict: ConceptDictionary
    ) -> tuple[str, str, str, str]:
        """
        Generate a complete authentication exchange pair.

        Args:
            name_id: Concept ID for the name
            company_id: Concept ID for the company
            topic_id: Concept ID for the topic
            language_pack: Language pack to use
            concept_dict: Concept dictionary

        Returns:
            Tuple of (english_client, synthetic_client, english_server, synthetic_server)
        """
        variables = {
            'name': name_id,
            'company': company_id,
            'topic': topic_id
        }

        # Get English versions
        name = concept_dict.get_concept(name_id).term
        company = concept_dict.get_concept(company_id).term
        topic = concept_dict.get_concept(topic_id).term

        english_client = f"Checking in, this is {name} with {company} talking about {topic}"
        english_server = f"Information received, confirmed you are {name} with {company} talking about {topic}"

        # Generate synthetic versions
        synthetic_client = self.translate_sentence(
            'authentication_client',
            variables,
            language_pack,
            concept_dict
        )

        synthetic_server = self.translate_sentence(
            'authentication_server',
            variables,
            language_pack,
            concept_dict
        )

        return english_client, synthetic_client, english_server, synthetic_server
