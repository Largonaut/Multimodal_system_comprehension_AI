"""
GREMLIN Model Evaluation - Quality Metrics

SECURITY COMPLIANCE:
- Zero hub dependencies
- Pure Python metrics calculation
- No external APIs or telemetry
- Local file operations only

Evaluates trained GREMLIN model on:
1. Translation accuracy (custom BLEU-like metric)
2. Compression ratio (character reduction)
3. Coverage (no English leakage)
4. Fluency (qualitative assessment)

Author: GREMLIN Team
License: MIT
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
from collections import Counter

# Import custom modules
from gemma_model import GemmaForCausalLM
from gremlin_tokenizer_wrapper import GremlinTokenizer


class GREMLINEvaluator:
    """
    Evaluates GREMLIN model quality and performance.
    """

    def __init__(
        self,
        model: GemmaForCausalLM,
        tokenizer: GremlinTokenizer,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained GREMLIN model
            tokenizer: GREMLIN tokenizer
            device: Device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def calculate_compression_ratio(self, english: str, gremlin: str) -> float:
        """
        Calculate character reduction ratio.

        Args:
            english: Original English text
            gremlin: GREMLIN compressed text

        Returns:
            Compression ratio (0.0 to 1.0, higher is better)
        """
        english_len = len(english)
        gremlin_len = len(gremlin)

        if english_len == 0:
            return 0.0

        reduction = (english_len - gremlin_len) / english_len
        return max(0.0, reduction)  # Clamp to 0 if expansion occurred

    def calculate_bleu_score(self, reference: str, hypothesis: str, n: int = 4) -> float:
        """
        Calculate BLEU-like score for translation quality.

        Simplified BLEU implementation (no external dependencies).

        Args:
            reference: Ground truth GREMLIN text
            hypothesis: Model-generated GREMLIN text
            n: Maximum n-gram size (default 4)

        Returns:
            BLEU score (0.0 to 1.0)
        """
        # Tokenize into characters (since GREMLIN uses Unicode characters)
        ref_tokens = list(reference)
        hyp_tokens = list(hypothesis)

        # Calculate precision for each n-gram size
        precisions = []

        for i in range(1, n + 1):
            # Get n-grams
            ref_ngrams = self._get_ngrams(ref_tokens, i)
            hyp_ngrams = self._get_ngrams(hyp_tokens, i)

            # Calculate precision
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue

            # Count matches
            matches = 0
            for ngram in hyp_ngrams:
                if ngram in ref_ngrams:
                    matches += 1
                    ref_ngrams.remove(ngram)  # Each reference n-gram used once

            precision = matches / len(hyp_ngrams)
            precisions.append(precision)

        # Brevity penalty
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)

        if hyp_len > ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            log_precisions = [np.log(p) for p in precisions]
            geo_mean = np.exp(np.mean(log_precisions))
        else:
            geo_mean = 0.0

        bleu = bp * geo_mean

        return bleu

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Extract n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def check_coverage(self, gremlin: str) -> bool:
        """
        Check if GREMLIN text contains English leakage.

        Args:
            gremlin: GREMLIN compressed text

        Returns:
            True if no leakage detected, False otherwise
        """
        # Simple heuristic: check for common English words
        # In GREMLIN, English should be fully compressed
        common_words = [
            "the", "and", "is", "in", "to", "of", "a", "for", "on", "with",
            "this", "that", "by", "from", "at", "as", "an", "be", "are"
        ]

        gremlin_lower = gremlin.lower()

        for word in common_words:
            # Check for word boundaries
            if f" {word} " in gremlin_lower or gremlin_lower.startswith(f"{word} ") or gremlin_lower.endswith(f" {word}"):
                return False  # Leakage detected

        return True  # No leakage

    def generate_translation(
        self,
        english: str,
        max_length: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate GREMLIN translation for English text.

        Args:
            english: Input English text
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Generated GREMLIN text
        """
        # Format as instruction
        prompt = f"Translate from English to GREMLIN:\nEnglish: {english}\nGREMLIN:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
            )

        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract GREMLIN part (after "GREMLIN:")
        if "GREMLIN:" in output_text:
            gremlin_text = output_text.split("GREMLIN:")[-1].strip()
        else:
            gremlin_text = output_text

        return gremlin_text

    def evaluate_sample(
        self,
        english: str,
        reference_gremlin: str,
    ) -> Dict[str, float]:
        """
        Evaluate a single English → GREMLIN sample.

        Args:
            english: English input
            reference_gremlin: Ground truth GREMLIN

        Returns:
            Dictionary of metrics
        """
        # Generate translation
        hypothesis_gremlin = self.generate_translation(english)

        # Calculate metrics
        metrics = {
            "bleu_score": self.calculate_bleu_score(reference_gremlin, hypothesis_gremlin),
            "compression_ratio_ref": self.calculate_compression_ratio(english, reference_gremlin),
            "compression_ratio_hyp": self.calculate_compression_ratio(english, hypothesis_gremlin),
            "coverage": 1.0 if self.check_coverage(hypothesis_gremlin) else 0.0,
        }

        return metrics

    def evaluate_dataset(
        self,
        test_file: str,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate on test dataset.

        Args:
            test_file: Path to test JSONL file
            num_samples: Number of samples to evaluate

        Returns:
            Aggregated metrics
        """
        print(f"\n📊 Evaluating on {num_samples} samples from {Path(test_file).name}")

        all_metrics = []

        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Evaluating")):
                if i >= num_samples:
                    break

                # Parse sample
                data = json.loads(line)

                # Extract English and GREMLIN
                # Assuming original corpus format
                if "input" in data and "output" in data:
                    english = data["input"]
                    reference = data["output"]
                else:
                    # Skip if wrong format
                    continue

                # Evaluate
                try:
                    metrics = self.evaluate_sample(english, reference)
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"\n⚠️  Error evaluating sample {i}: {e}")
                    continue

        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

        return aggregated

    def print_evaluation_report(self, metrics: Dict[str, float]):
        """Print formatted evaluation report."""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)

        print(f"\n📈 Translation Quality:")
        print(f"   BLEU Score:        {metrics.get('bleu_score_mean', 0.0):.4f} ± {metrics.get('bleu_score_std', 0.0):.4f}")

        print(f"\n🗜️  Compression Performance:")
        print(f"   Reference Ratio:   {metrics.get('compression_ratio_ref_mean', 0.0):.2%} ± {metrics.get('compression_ratio_ref_std', 0.0):.2%}")
        print(f"   Hypothesis Ratio:  {metrics.get('compression_ratio_hyp_mean', 0.0):.2%} ± {metrics.get('compression_ratio_hyp_std', 0.0):.2%}")

        print(f"\n✓ Coverage:")
        print(f"   No Leakage:        {metrics.get('coverage_mean', 0.0):.2%}")

        print("=" * 70)

    def interactive_demo(self):
        """Interactive translation demo."""
        print("\n" + "=" * 70)
        print("INTERACTIVE GREMLIN TRANSLATOR")
        print("=" * 70)
        print("Enter English text to translate to GREMLIN")
        print("Type 'quit' to exit")
        print("=" * 70)

        while True:
            english = input("\nEnglish: ").strip()

            if english.lower() in ['quit', 'exit', 'q']:
                break

            if not english:
                continue

            # Generate translation
            gremlin = self.generate_translation(english)

            # Calculate metrics
            compression = self.calculate_compression_ratio(english, gremlin)
            coverage = self.check_coverage(gremlin)

            print(f"\nGREMLIN: {gremlin}")
            print(f"\nMetrics:")
            print(f"  Compression: {compression:.2%}")
            print(f"  Coverage:    {'✓ No leakage' if coverage else '✗ English detected'}")


def main():
    """Main execution function."""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║               GREMLIN MODEL EVALUATION                             ║
    ║                                                                    ║
    ║  Quality Metrics:                                                  ║
    ║    • Translation accuracy (BLEU score)                             ║
    ║    • Compression ratio (55-80% target)                             ║
    ║    • Coverage (0% English leakage)                                 ║
    ║    • Fluency (qualitative)                                         ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    model_path = "F:/dev/GREMLIN_Claude_Code_Web_track/models/gemma-2-9b-gremlin"
    tokenizer_path = "F:/dev/GREMLIN_Claude_Code_Web_track/models/tokenizer/gremlin_tokenizer.json"
    test_file = "F:/dev/GREMLIN_Claude_Code_Web_track/training_data/gremlin_god_mode_corpus.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    # Load tokenizer
    print(f"\n📚 Loading tokenizer...")
    tokenizer = GremlinTokenizer(tokenizer_path)

    # Load model (placeholder - in production, load trained model)
    print(f"\n📦 Loading trained model...")
    # model = GemmaForCausalLM.load(model_path)

    # Initialize evaluator
    # evaluator = GREMLINEvaluator(model, tokenizer, device)

    # Run evaluation
    # metrics = evaluator.evaluate_dataset(test_file, num_samples=100)
    # evaluator.print_evaluation_report(metrics)

    # Interactive demo
    # evaluator.interactive_demo()

    print("\n✅ Evaluation pipeline ready!")
    print("   (Uncomment model loading to run full evaluation)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
