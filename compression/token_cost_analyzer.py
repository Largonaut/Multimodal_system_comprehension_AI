"""
Token Cost Analyzer for Unicode Characters
Analyzes Unicode characters by LLM tokenizer cost to find optimal 1-token chars.
"""

from typing import List, Tuple, Dict
from collections import defaultdict
import json
from pathlib import Path


class TokenCostAnalyzer:
    """Analyze Unicode characters by tokenizer cost."""

    def __init__(self):
        """Initialize the analyzer."""
        self.single_token_chars = []
        self.tokenizer = None

    def load_tokenizer(self, model_name: str = "gpt2"):
        """
        Load a tokenizer to test token costs.

        Args:
            model_name: Name of the model tokenizer (gpt2, gpt-3.5-turbo, etc.)
        """
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✓ Loaded {model_name} tokenizer")
            return True
        except ImportError:
            print("⚠️  transformers not available, using fallback analysis")
            return False
        except Exception as e:
            print(f"⚠️  Could not load tokenizer: {e}")
            return False

    def get_token_cost(self, text: str) -> int:
        """
        Get token cost for a string.

        Args:
            text: Text to analyze

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        else:
            # Fallback: assume ASCII = 1 token, others = 2-3
            cost = 0
            for char in text:
                if ord(char) < 128:
                    cost += 1
                elif ord(char) < 0x4E00:  # Before CJK
                    cost += 2
                else:
                    cost += 3
            return cost

    def analyze_unicode_blocks(self, max_codepoint: int = 0x10000) -> Dict[str, List[str]]:
        """
        Analyze Unicode characters and categorize by token cost.

        Args:
            max_codepoint: Maximum Unicode codepoint to analyze

        Returns:
            Dict mapping token_cost -> list of characters
        """
        print(f"Analyzing Unicode characters up to U+{max_codepoint:04X}...")

        cost_buckets = defaultdict(list)
        single_token_count = 0

        for codepoint in range(32, max_codepoint):  # Skip control chars
            try:
                char = chr(codepoint)
                # Skip certain ranges
                if codepoint in range(0xD800, 0xE000):  # Surrogates
                    continue
                if codepoint in range(0xFDD0, 0xFDF0):  # Non-characters
                    continue

                cost = self.get_token_cost(char)
                cost_buckets[cost].append(char)

                if cost == 1:
                    single_token_count += 1

            except Exception:
                continue

        print(f"\n✓ Analysis complete!")
        print(f"  1-token chars: {len(cost_buckets[1]):,}")
        print(f"  2-token chars: {len(cost_buckets[2]):,}")
        print(f"  3+ token chars: {sum(len(cost_buckets[c]) for c in cost_buckets if c > 2):,}")

        return dict(cost_buckets)

    def get_best_single_token_chars(self, n: int = 1000) -> List[str]:
        """
        Get the N best single-token Unicode characters.

        Prioritizes:
        - Visually distinct
        - Cross-script diversity
        - Common in existing text

        Args:
            n: Number of characters to return

        Returns:
            List of single-token characters
        """
        cost_buckets = self.analyze_unicode_blocks()
        single_token = cost_buckets.get(1, [])

        # Priority blocks for diversity
        priority_blocks = [
            (0x0030, 0x0039),  # Digits
            (0x0041, 0x005A),  # Latin uppercase
            (0x0061, 0x007A),  # Latin lowercase
            (0x00C0, 0x00FF),  # Latin Extended-A
            (0x0391, 0x03C9),  # Greek
            (0x0410, 0x044F),  # Cyrillic
            (0x05D0, 0x05EA),  # Hebrew
            (0x0600, 0x06FF),  # Arabic
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x4F00),  # CJK sample
        ]

        selected = []

        # First, get priority block chars
        for start, end in priority_blocks:
            for char in single_token:
                if start <= ord(char) <= end:
                    selected.append(char)
                    if len(selected) >= n:
                        return selected[:n]

        # Fill remaining with any single-token chars
        for char in single_token:
            if char not in selected:
                selected.append(char)
                if len(selected) >= n:
                    break

        return selected[:n]

    def generate_2char_combos(self, base_chars: List[str], n: int = 220000) -> List[str]:
        """
        Generate 2-character combinations from base characters.

        Args:
            base_chars: List of base characters to combine
            n: Number of combinations to generate

        Returns:
            List of 2-character combinations
        """
        combos = []
        for c1 in base_chars:
            for c2 in base_chars:
                combos.append(c1 + c2)
                if len(combos) >= n:
                    return combos
        return combos

    def save_analysis(self, output_path: Path):
        """Save analysis results to JSON."""
        cost_buckets = self.analyze_unicode_blocks()

        # Convert to serializable format
        data = {
            'single_token_count': len(cost_buckets.get(1, [])),
            'two_token_count': len(cost_buckets.get(2, [])),
            'sample_single_token': cost_buckets.get(1, [])[:100],
            'best_chars': self.get_best_single_token_chars(1000)
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved analysis to {output_path}")


def main():
    """Run token cost analysis."""
    print("=" * 60)
    print("Unicode Token Cost Analyzer")
    print("=" * 60)
    print()

    analyzer = TokenCostAnalyzer()
    analyzer.load_tokenizer("gpt2")

    # Get best single-token chars
    best_chars = analyzer.get_best_single_token_chars(500)
    print(f"\n✓ Selected {len(best_chars)} optimal 1-token characters")
    print(f"Sample: {''.join(best_chars[:50])}")

    # Generate 2-char combos
    combos = analyzer.generate_2char_combos(best_chars, 220000)
    print(f"\n✓ Generated {len(combos):,} 2-character combinations")
    print(f"Sample: {', '.join(combos[:10])}")

    # Calculate coverage
    max_coverage = len(best_chars) + len(combos)
    print(f"\n✓ Total coverage: {max_coverage:,} unique codes")
    print(f"  1-char codes: {len(best_chars):,}")
    print(f"  2-char codes: {len(combos):,}")

    # Save results
    output_path = Path(__file__).parent / "token_analysis.json"
    analyzer.save_analysis(output_path)


if __name__ == '__main__':
    main()
