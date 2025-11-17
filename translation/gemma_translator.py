"""
Gemma-Powered Translator
Uses Gemma 2 9B for intelligent translation between English and synthetic languages.
"""

from pathlib import Path
from typing import Optional, Dict, List
import re

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from core import ConceptDictionary, LanguagePack, GrammarEngine


class GemmaTranslator:
    """
    AI-powered translator using Gemma 2 9B.

    Translates between English and synthetic languages with context awareness.
    """

    def __init__(
        self,
        language_pack: LanguagePack,
        concept_dict: ConceptDictionary,
        gemma_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_ai: bool = True
    ):
        """
        Initialize translator.

        Args:
            language_pack: Language pack to use
            concept_dict: Concept dictionary
            gemma_model_path: Path to Gemma 2 9B model
            device: Device to run on
            use_ai: Use AI model (False = dictionary-only mode)
        """
        self.pack = language_pack
        self.concept_dict = concept_dict
        self.grammar = GrammarEngine()
        self.device = device
        self.use_ai = use_ai and HAS_TRANSFORMERS

        self.model = None
        self.tokenizer = None

        if self.use_ai:
            self._load_model(gemma_model_path)
        else:
            print("🔧 Running in dictionary-only mode (no AI)")

    def _load_model(self, model_path: Optional[str]):
        """Load Gemma 2 model."""
        if model_path and Path(model_path).exists():
            print(f"📦 Loading Gemma 2 from {model_path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                self.model.eval()
                print(f"✅ Gemma 2 loaded on {self.device}")
            except Exception as e:
                print(f"⚠️  Could not load Gemma 2: {e}")
                print("🔧 Falling back to dictionary-only mode")
                self.use_ai = False
        else:
            print("ℹ️  No Gemma model path provided, using dictionary-only mode")
            self.use_ai = False

    def translate_to_synthetic(
        self,
        english_text: str,
        template_name: str = None,
        variables: Dict[str, str] = None
    ) -> str:
        """
        Translate English to synthetic language.

        Args:
            english_text: English input
            template_name: Optional template to use
            variables: Optional variable mappings

        Returns:
            Synthetic language text
        """
        if template_name and variables:
            # Use template-based translation
            return self.grammar.translate_sentence(
                template_name,
                variables,
                self.pack,
                self.concept_dict
            )

        # Dictionary-based translation (word-by-word)
        return self._dictionary_translate(english_text, to_synthetic=True)

    def translate_to_english(
        self,
        synthetic_text: str,
        reverse_map: Dict[str, str] = None
    ) -> str:
        """
        Translate synthetic language to English.

        Args:
            synthetic_text: Synthetic language input
            reverse_map: Optional reverse mapping (synthetic word -> concept_id)

        Returns:
            English text
        """
        if reverse_map:
            # Use provided reverse mapping
            words = synthetic_text.split()
            english_words = []

            for word in words:
                concept_id = reverse_map.get(word)
                if concept_id:
                    concept = self.concept_dict.get_concept(concept_id)
                    if concept:
                        english_words.append(concept.term)
                else:
                    english_words.append(word)  # Keep unknown words

            return ' '.join(english_words)

        # Try to reverse lookup from language pack
        return self._dictionary_translate(synthetic_text, to_synthetic=False)

    def _dictionary_translate(self, text: str, to_synthetic: bool = True) -> str:
        """
        Simple dictionary-based translation.

        Args:
            text: Input text
            to_synthetic: Direction (True = English->Synthetic, False = Synthetic->English)

        Returns:
            Translated text
        """
        if to_synthetic:
            # English -> Synthetic (simplified word-by-word)
            words = text.lower().replace(',', '').split()
            synthetic_words = []

            for word in words:
                # Search for concept
                matches = self.concept_dict.search_by_term(word)
                if matches:
                    concept = matches[0]
                    syn_word = self.pack.get_word_for_concept(concept.id)
                    if syn_word:
                        synthetic_words.append(syn_word)
                else:
                    synthetic_words.append(word)  # Keep unknown words

            return ' '.join(synthetic_words)
        else:
            # Synthetic -> English (lookup each word)
            words = text.split()
            english_words = []

            for word in words:
                # Search in all concept pools
                found = False
                for concept_id, pool in self.pack.word_pools.items():
                    if word in pool.words:
                        concept = self.concept_dict.get_concept(concept_id)
                        if concept:
                            english_words.append(concept.term)
                            found = True
                            break

                if not found:
                    english_words.append(word)

            return ' '.join(english_words)

    def generate_with_ai(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using Gemma 2 (for advanced features).

        Args:
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Generated text
        """
        if not self.use_ai or self.model is None:
            return ""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def create_reverse_map(self, synthetic_text: str) -> Dict[str, str]:
        """
        Create reverse mapping for synthetic text.

        Args:
            synthetic_text: Synthetic language text

        Returns:
            Dict mapping synthetic words to concept_ids
        """
        reverse_map = {}
        words = synthetic_text.split()

        for word in words:
            for concept_id, pool in self.pack.word_pools.items():
                if word in pool.words:
                    reverse_map[word] = concept_id
                    break

        return reverse_map

    def __repr__(self):
        mode = "AI" if self.use_ai else "Dictionary"
        return f"GemmaTranslator(mode={mode}, device={self.device})"
