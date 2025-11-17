"""
GREMLIN Engine - Backend for Admin Console
Handles client/server logic in both simulated and network modes.
"""

import random
import time
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ConceptDictionary, LanguagePack, GrammarEngine


@dataclass
class Message:
    """Represents a message in the system."""
    timestamp: str
    direction: str  # 'client->server' or 'server->client'
    english: str
    synthetic: str
    sender: str  # 'client' or 'server'


class GremlinEngine:
    """
    Backend engine for GREMLIN demo.
    Supports both simulated and network modes.
    """

    def __init__(
        self,
        language_pack_path: Path,
        mode: str = "demo"
    ):
        """
        Initialize GREMLIN engine.

        Args:
            language_pack_path: Path to language pack
            mode: 'demo' (simulated) or 'network' (real)
        """
        self.mode = mode
        self.pack = LanguagePack.load(language_pack_path)
        self.concept_dict = ConceptDictionary()
        self.grammar = GrammarEngine()

        # Create translator for freeform messages
        from translation.gemma_translator import GemmaTranslator
        self.translator = GemmaTranslator(
            self.pack,
            self.concept_dict,
            use_ai=False  # Dictionary mode
        )

        self.messages: List[Message] = []
        self.packet_count = 0
        self.attack_count = 0

        # Callbacks for UI updates
        self.on_message: Optional[Callable] = None
        self.on_stats_update: Optional[Callable] = None

    def send_authentication(self, name: str = None, company: str = None, topic: str = None) -> Message:
        """
        Send authentication message from client.

        Args:
            name: Name (random if None)
            company: Company (random if None)
            topic: Topic (random if None)

        Returns:
            Message object
        """
        # Pick random values if not provided
        if not name:
            names = self.concept_dict.get_concepts_by_category('variables_names')
            name_concept = random.choice(names)
            name = name_concept.term
            name_id = name_concept.id
        else:
            matches = self.concept_dict.search_by_term(name)
            name_id = matches[0].id if matches else None

        if not company:
            companies = self.concept_dict.get_concepts_by_category('variables_companies')
            company_concept = random.choice(companies)
            company = company_concept.term
            company_id = company_concept.id
        else:
            matches = self.concept_dict.search_by_term(company)
            company_id = matches[0].id if matches else None

        if not topic:
            topics = self.concept_dict.get_concepts_by_category('variables_topics')
            topic_concept = random.choice(topics)
            topic = topic_concept.term
            topic_id = topic_concept.id
        else:
            matches = self.concept_dict.search_by_term(topic)
            topic_id = matches[0].id if matches else None

        # Generate message
        variables = {
            'name': name_id,
            'company': company_id,
            'topic': topic_id
        }

        english = f"Checking in, this is {name} with {company} talking about {topic}"
        synthetic = self.grammar.translate_sentence(
            'authentication_client',
            variables,
            self.pack,
            self.concept_dict
        )

        msg = Message(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            direction='client->server',
            english=english,
            synthetic=synthetic,
            sender='client'
        )

        self.messages.append(msg)
        self.packet_count += 1

        if self.on_message:
            self.on_message(msg)

        # Generate server response
        self._generate_server_response(name, company, topic, name_id, company_id, topic_id)

        return msg

    def _generate_server_response(
        self,
        name: str,
        company: str,
        topic: str,
        name_id: str,
        company_id: str,
        topic_id: str
    ):
        """Generate server response to authentication."""
        variables = {
            'name': name_id,
            'company': company_id,
            'topic': topic_id
        }

        english = f"Information received, confirmed you are {name} with {company} talking about {topic}"
        synthetic = self.grammar.translate_sentence(
            'authentication_server',
            variables,
            self.pack,
            self.concept_dict
        )

        msg = Message(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            direction='server->client',
            english=english,
            synthetic=synthetic,
            sender='server'
        )

        self.messages.append(msg)
        self.packet_count += 1

        if self.on_message:
            self.on_message(msg)

    def simulate_attack(self) -> str:
        """
        Simulate a MITM attack attempt.

        Returns:
            Gibberish that attacker would see
        """
        self.attack_count += 1

        # Return random message's synthetic text
        if self.messages:
            return random.choice(self.messages).synthetic
        else:
            return "ポΉlڡ؜o]ӥTϑ エsπζϗ ギ؃ッ΂ϖCOボぜз"

    def get_stats(self) -> Dict:
        """Get current statistics."""
        pack_stats = self.pack.get_stats()

        return {
            'language_id': pack_stats['language_id'][:16] + '...',
            'total_words': pack_stats['total_words'],
            'words_used': pack_stats['total_used'],
            'words_remaining': pack_stats['words_remaining'],
            'usage_percent': pack_stats['usage_percentage'],
            'packet_count': self.packet_count,
            'attack_count': self.attack_count,
            'message_count': len(self.messages),
            'estimated_rounds': int(pack_stats['words_remaining'] / 8) if pack_stats['words_remaining'] > 0 else 0
        }

    def generate_new_pack(
        self,
        words_per_concept: int = 1000,
        output_path: Path = None
    ) -> Path:
        """
        Generate a new language pack.

        Args:
            words_per_concept: Words to generate per concept
            output_path: Where to save (None = auto)

        Returns:
            Path to new pack
        """
        from core import WordGenerator

        wg = WordGenerator(
            min_length=4,
            max_length=12,
            use_blocks=['latin_basic', 'cyrillic', 'greek', 'arabic', 'hiragana', 'katakana']
        )

        new_pack = LanguagePack.generate(
            self.concept_dict,
            words_per_concept=words_per_concept,
            word_generator=wg
        )

        if output_path is None:
            output_path = Path(f"language_packs/language_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        new_pack.save(output_path)

        return output_path

    def rotate_language(self, new_pack_path: Path):
        """
        Rotate to a new language pack.

        Args:
            new_pack_path: Path to new language pack
        """
        self.pack = LanguagePack.load(new_pack_path)
        self.messages.clear()
        self.packet_count = 0

        if self.on_stats_update:
            self.on_stats_update()

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get recent messages."""
        return self.messages[-limit:]
