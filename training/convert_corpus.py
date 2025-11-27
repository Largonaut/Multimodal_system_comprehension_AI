"""
GREMLIN Corpus Converter - Instruction Format

SECURITY COMPLIANCE:
- Zero hub dependencies
- Pure Python file operations
- Streaming architecture (memory-efficient for 2.3GB files)
- No network calls

Converts GREMLIN corpus from translation pairs to instruction-following format.

Input format:
{"source": "brown", "input": "English text", "output": "GREMLIN text"}

Output format:
{"text": "Translate from English to GREMLIN:\nEnglish: ...\nGREMLIN: ...<|endoftext|>"}

Author: GREMLIN Team
License: MIT
"""

import json
import random
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm


class CorpusConverter:
    """
    Converts GREMLIN corpus to instruction-tuning format.

    Streams large files (2.3GB) without loading into memory.
    """

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        train_split: float = 0.95,
        val_split: float = 0.05,
        seed: int = 42,
    ):
        """
        Initialize corpus converter.

        Args:
            input_path: Path to gremlin_god_mode_corpus.jsonl
            output_dir: Directory to save converted files
            train_split: Fraction for training set (default 95%)
            val_split: Fraction for validation set (default 5%)
            seed: Random seed for reproducibility
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output file paths
        self.train_output = self.output_dir / "gremlin_instruction_train.jsonl"
        self.val_output = self.output_dir / "gremlin_instruction_val.jsonl"

        # Statistics
        self.stats = {
            "total_lines": 0,
            "train_lines": 0,
            "val_lines": 0,
            "skipped_lines": 0,
            "source_counts": {},
        }

        # Set random seed
        random.seed(self.seed)

    def format_instruction(self, english: str, gremlin: str) -> str:
        """
        Format a training pair as instruction-following text.

        Args:
            english: English input text
            gremlin: GREMLIN compressed output

        Returns:
            Formatted instruction string
        """
        # Instruction-following format for causal LM training
        instruction = (
            f"Translate from English to GREMLIN:\n"
            f"English: {english}\n"
            f"GREMLIN: {gremlin}<|endoftext|>"
        )

        return instruction

    def count_lines(self) -> int:
        """
        Count total lines in input file.

        Returns:
            Number of lines
        """
        print(f"\n📊 Counting lines in {self.input_path.name}...")
        line_count = 0

        with open(self.input_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1

        print(f"   Total lines: {line_count:,}")
        return line_count

    def convert(self, sample_size: Optional[int] = None):
        """
        Convert corpus to instruction format.

        Args:
            sample_size: Optional - only convert first N samples (for testing)
        """
        print("=" * 70)
        print("GREMLIN CORPUS CONVERTER - INSTRUCTION FORMAT")
        print("=" * 70)
        print(f"Input:  {self.input_path}")
        print(f"Output: {self.output_dir}")
        print(f"Split:  {self.train_split*100:.1f}% train / {self.val_split*100:.1f}% val")
        if sample_size:
            print(f"Sample: First {sample_size:,} lines only")
        print("=" * 70)

        # Count lines
        total_lines = self.count_lines()
        if sample_size:
            total_lines = min(total_lines, sample_size)

        # Open output files
        train_file = open(self.train_output, 'w', encoding='utf-8')
        val_file = open(self.val_output, 'w', encoding='utf-8')

        # Stream and convert
        print(f"\n🔄 Converting corpus...")

        with open(self.input_path, 'r', encoding='utf-8') as input_file:
            for line_num, line in enumerate(tqdm(input_file, total=total_lines, desc="Converting")):
                # Stop if sample size reached
                if sample_size and line_num >= sample_size:
                    break

                try:
                    # Parse JSONL
                    data = json.loads(line.strip())

                    # Extract fields
                    source = data.get("source", "unknown")
                    english = data.get("input", "")
                    gremlin = data.get("output", "")

                    # Skip invalid entries
                    if not english or not gremlin:
                        self.stats["skipped_lines"] += 1
                        continue

                    # Format as instruction
                    instruction = self.format_instruction(english, gremlin)

                    # Create output entry
                    output_entry = {"text": instruction}

                    # Randomly assign to train or val
                    if random.random() < self.train_split:
                        train_file.write(json.dumps(output_entry) + '\n')
                        self.stats["train_lines"] += 1
                    else:
                        val_file.write(json.dumps(output_entry) + '\n')
                        self.stats["val_lines"] += 1

                    # Track source statistics
                    if source not in self.stats["source_counts"]:
                        self.stats["source_counts"][source] = 0
                    self.stats["source_counts"][source] += 1

                    self.stats["total_lines"] += 1

                except json.JSONDecodeError as e:
                    print(f"\n⚠️  Skipping invalid JSON at line {line_num}: {e}")
                    self.stats["skipped_lines"] += 1
                    continue
                except Exception as e:
                    print(f"\n⚠️  Error at line {line_num}: {e}")
                    self.stats["skipped_lines"] += 1
                    continue

        # Close files
        train_file.close()
        val_file.close()

        # Print statistics
        self.print_statistics()

    def print_statistics(self):
        """Print conversion statistics."""
        print("\n" + "=" * 70)
        print("CONVERSION COMPLETE")
        print("=" * 70)

        print(f"\n📊 Statistics:")
        print(f"   Total processed:  {self.stats['total_lines']:>12,}")
        print(f"   Training set:     {self.stats['train_lines']:>12,} ({self.stats['train_lines']/self.stats['total_lines']*100:.1f}%)")
        print(f"   Validation set:   {self.stats['val_lines']:>12,} ({self.stats['val_lines']/self.stats['total_lines']*100:.1f}%)")
        print(f"   Skipped:          {self.stats['skipped_lines']:>12,}")

        print(f"\n📚 Source Distribution:")
        for source, count in sorted(self.stats["source_counts"].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.stats["total_lines"]) * 100
            print(f"   {source:<15} {count:>12,} ({percentage:>5.1f}%)")

        # File sizes
        train_size = self.train_output.stat().st_size / (1024**3)  # GB
        val_size = self.val_output.stat().st_size / (1024**3)

        print(f"\n💾 Output Files:")
        print(f"   Training:   {self.train_output}")
        print(f"               {train_size:.2f} GB")
        print(f"   Validation: {self.val_output}")
        print(f"               {val_size:.2f} GB")

        print("=" * 70)

    def sample_examples(self, num_examples: int = 5):
        """
        Print sample examples from converted corpus.

        Args:
            num_examples: Number of examples to print
        """
        print(f"\n📝 Sample Examples (first {num_examples}):")
        print("-" * 70)

        with open(self.train_output, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break

                data = json.loads(line)
                text = data["text"]

                # Truncate if too long
                if len(text) > 200:
                    text = text[:200] + "..."

                print(f"\nExample {i+1}:")
                print(text)

        print("-" * 70)


def main():
    """Main execution function."""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           GREMLIN CORPUS CONVERTER - INSTRUCTION FORMAT            ║
    ║                                                                    ║
    ║  Transforms 14.9M translation pairs into instruction-tuning format ║
    ║  Streaming architecture - handles 2.3GB files efficiently          ║
    ║  Security: Pure Python, no dependencies, no network calls          ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    input_path = "F:/dev/GREMLIN_Claude_Code_Web_track/training_data/gremlin_god_mode_corpus.jsonl"
    output_dir = "F:/dev/GREMLIN_Claude_Code_Web_track/training_data"

    # Initialize converter
    converter = CorpusConverter(
        input_path=input_path,
        output_dir=output_dir,
        train_split=0.95,
        val_split=0.05,
        seed=42,
    )

    # Option to test with sample first
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n🧪 TEST MODE: Converting first 10,000 samples only")
        converter.convert(sample_size=10000)
    else:
        # Full conversion
        converter.convert()

    # Show examples
    converter.sample_examples(num_examples=5)

    print("\n✅ Conversion complete! Ready for training.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
