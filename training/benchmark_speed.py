"""
GREMLIN Performance Benchmarking

SECURITY COMPLIANCE:
- Zero hub dependencies
- Pure PyTorch profiling
- Local measurements only
- No telemetry

Benchmarks GREMLIN model to verify compute savings:
1. Inference speed (tokens/sec)
2. Context window efficiency (how much more content fits)
3. Memory usage (KV cache size)
4. Latency comparison (English vs GREMLIN)

Author: GREMLIN Team
License: MIT
"""

import torch
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

# Import custom modules
from gemma_model import GemmaForCausalLM
from gremlin_tokenizer_wrapper import GremlinTokenizer


class GREMLINBenchmark:
    """
    Benchmarks GREMLIN model performance.

    Measures inference speed, memory usage, and context efficiency.
    """

    def __init__(
        self,
        gremlin_model: GemmaForCausalLM,
        gremlin_tokenizer: GremlinTokenizer,
        baseline_model: GemmaForCausalLM = None,
        baseline_tokenizer = None,
        device: str = "cuda",
    ):
        """
        Initialize benchmark.

        Args:
            gremlin_model: Trained GREMLIN model
            gremlin_tokenizer: GREMLIN tokenizer
            baseline_model: Optional baseline Gemma model for comparison
            baseline_tokenizer: Optional baseline tokenizer
            device: Device for inference
        """
        self.gremlin_model = gremlin_model
        self.gremlin_tokenizer = gremlin_tokenizer
        self.baseline_model = baseline_model
        self.baseline_tokenizer = baseline_tokenizer
        self.device = device

        self.gremlin_model.to(self.device)
        self.gremlin_model.eval()

        if self.baseline_model:
            self.baseline_model.to(self.device)
            self.baseline_model.eval()

    def measure_inference_speed(
        self,
        model: GemmaForCausalLM,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 5,
    ) -> Dict[str, float]:
        """
        Measure inference speed.

        Args:
            model: Model to benchmark
            tokenizer: Tokenizer to use
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            num_runs: Number of benchmark runs

        Returns:
            Dictionary with speed metrics
        """
        # Warmup
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            _ = model.generate(input_ids, max_length=10)

        # Benchmark
        latencies = []
        token_counts = []

        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            start_time = time.time()

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=input_ids.size(1) + max_new_tokens,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None

            end_time = time.time()

            latency = end_time - start_time
            num_tokens = output_ids.size(1) - input_ids.size(1)

            latencies.append(latency)
            token_counts.append(num_tokens)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        avg_tokens = np.mean(token_counts)
        tokens_per_sec = avg_tokens / avg_latency

        return {
            "latency_mean": avg_latency,
            "latency_std": np.std(latencies),
            "tokens_generated": avg_tokens,
            "tokens_per_sec": tokens_per_sec,
        }

    def measure_context_efficiency(
        self,
        sample_texts: List[str],
        max_context_length: int = 8192,
    ) -> Dict[str, float]:
        """
        Measure how much content fits in context window.

        Args:
            sample_texts: List of sample texts
            max_context_length: Maximum context window size

        Returns:
            Dictionary with efficiency metrics
        """
        gremlin_token_counts = []
        baseline_token_counts = []
        compression_ratios = []

        for text in sample_texts:
            # Tokenize with GREMLIN
            gremlin_tokens = self.gremlin_tokenizer.encode(text, add_special_tokens=False)
            gremlin_count = len(gremlin_tokens)
            gremlin_token_counts.append(gremlin_count)

            # Tokenize with baseline (if available)
            if self.baseline_tokenizer:
                baseline_tokens = self.baseline_tokenizer.encode(text, add_special_tokens=False)
                baseline_count = len(baseline_tokens)
                baseline_token_counts.append(baseline_count)

                # Compression ratio
                if baseline_count > 0:
                    ratio = (baseline_count - gremlin_count) / baseline_count
                    compression_ratios.append(ratio)

        # Calculate efficiency multiplier
        if baseline_token_counts:
            avg_gremlin = np.mean(gremlin_token_counts)
            avg_baseline = np.mean(baseline_token_counts)
            efficiency_multiplier = avg_baseline / avg_gremlin if avg_gremlin > 0 else 1.0

            # How many more texts fit in context
            texts_in_context_baseline = max_context_length / avg_baseline
            texts_in_context_gremlin = max_context_length / avg_gremlin

            return {
                "avg_gremlin_tokens": avg_gremlin,
                "avg_baseline_tokens": avg_baseline,
                "efficiency_multiplier": efficiency_multiplier,
                "compression_ratio": np.mean(compression_ratios),
                "texts_in_context_baseline": texts_in_context_baseline,
                "texts_in_context_gremlin": texts_in_context_gremlin,
            }
        else:
            return {
                "avg_gremlin_tokens": np.mean(gremlin_token_counts),
                "efficiency_multiplier": 1.0,
            }

    def measure_memory_usage(
        self,
        model: GemmaForCausalLM,
        tokenizer,
        prompt: str,
        sequence_length: int = 512,
    ) -> Dict[str, float]:
        """
        Measure GPU memory usage during inference.

        Args:
            model: Model to benchmark
            tokenizer: Tokenizer to use
            prompt: Input prompt
            sequence_length: Sequence length to process

        Returns:
            Dictionary with memory metrics
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Encode input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=sequence_length, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        # Measure baseline memory
        baseline_memory = torch.cuda.memory_allocated() / (1024**2)  # MB

        # Forward pass
        with torch.no_grad():
            _ = model(input_ids)

        # Measure peak memory
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        # Memory for this operation
        operation_memory = peak_memory - baseline_memory

        return {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "operation_memory_mb": operation_memory,
        }

    def run_full_benchmark(
        self,
        test_prompts: List[str],
        output_path: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Run complete benchmark suite.

        Args:
            test_prompts: List of test prompts
            output_path: Optional path to save results

        Returns:
            Complete benchmark results
        """
        print("=" * 70)
        print("GREMLIN PERFORMANCE BENCHMARK")
        print("=" * 70)

        results = {}

        # 1. Inference Speed
        print("\n🚀 Measuring inference speed...")
        speed_results = []

        for prompt in tqdm(test_prompts[:5], desc="Speed test"):
            metrics = self.measure_inference_speed(
                self.gremlin_model,
                self.gremlin_tokenizer,
                prompt,
            )
            speed_results.append(metrics)

        results["inference_speed"] = {
            "tokens_per_sec_mean": np.mean([r["tokens_per_sec"] for r in speed_results]),
            "tokens_per_sec_std": np.std([r["tokens_per_sec"] for r in speed_results]),
            "latency_mean": np.mean([r["latency_mean"] for r in speed_results]),
        }

        # 2. Context Efficiency
        print("\n📊 Measuring context window efficiency...")
        efficiency = self.measure_context_efficiency(test_prompts)
        results["context_efficiency"] = efficiency

        # 3. Memory Usage
        if torch.cuda.is_available():
            print("\n💾 Measuring memory usage...")
            memory = self.measure_memory_usage(
                self.gremlin_model,
                self.gremlin_tokenizer,
                test_prompts[0],
            )
            results["memory_usage"] = memory

        # Print report
        self.print_benchmark_report(results)

        # Save results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {output_path}")

        return results

    def print_benchmark_report(self, results: Dict[str, any]):
        """Print formatted benchmark report."""
        print("\n" + "=" * 70)
        print("BENCHMARK REPORT")
        print("=" * 70)

        # Inference Speed
        if "inference_speed" in results:
            speed = results["inference_speed"]
            print(f"\n🚀 Inference Speed:")
            print(f"   Tokens/sec:  {speed['tokens_per_sec_mean']:.2f} ± {speed['tokens_per_sec_std']:.2f}")
            print(f"   Latency:     {speed['latency_mean']*1000:.1f} ms")

        # Context Efficiency
        if "context_efficiency" in results:
            eff = results["context_efficiency"]
            print(f"\n📊 Context Window Efficiency:")

            if "avg_baseline_tokens" in eff:
                print(f"   GREMLIN tokens:    {eff['avg_gremlin_tokens']:.1f}")
                print(f"   Baseline tokens:   {eff['avg_baseline_tokens']:.1f}")
                print(f"   Efficiency:        {eff['efficiency_multiplier']:.2f}x")
                print(f"   Compression:       {eff['compression_ratio']:.1%}")
                print(f"\n   Texts in 8K context:")
                print(f"     Baseline:  {eff['texts_in_context_baseline']:.1f} texts")
                print(f"     GREMLIN:   {eff['texts_in_context_gremlin']:.1f} texts")
                print(f"     Gain:      {eff['texts_in_context_gremlin'] - eff['texts_in_context_baseline']:.1f} more texts")
            else:
                print(f"   Avg tokens: {eff['avg_gremlin_tokens']:.1f}")

        # Memory Usage
        if "memory_usage" in results:
            mem = results["memory_usage"]
            if "error" not in mem:
                print(f"\n💾 Memory Usage:")
                print(f"   Baseline:    {mem['baseline_memory_mb']:.1f} MB")
                print(f"   Peak:        {mem['peak_memory_mb']:.1f} MB")
                print(f"   Operation:   {mem['operation_memory_mb']:.1f} MB")

        print("=" * 70)


def main():
    """Main execution function."""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║              GREMLIN PERFORMANCE BENCHMARKING                      ║
    ║                                                                    ║
    ║  Measurements:                                                     ║
    ║    • Inference speed (tokens/sec)                                  ║
    ║    • Context efficiency (2-4x target)                              ║
    ║    • Memory usage (KV cache)                                       ║
    ║    • Latency comparison                                            ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    model_path = "F:/dev/GREMLIN_Claude_Code_Web_track/models/gemma-2-9b-gremlin"
    tokenizer_path = "F:/dev/GREMLIN_Claude_Code_Web_track/models/tokenizer/gremlin_tokenizer.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    # Load tokenizer
    print(f"\n📚 Loading tokenizer...")
    tokenizer = GremlinTokenizer(tokenizer_path)

    # Load model (placeholder)
    print(f"\n📦 Loading model...")
    # model = GemmaForCausalLM.load(model_path)

    # Test prompts
    test_prompts = [
        "Artificial intelligence is transforming how we live and work.",
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning was the Word, and the Word was with God.",
        "Machine learning models process data to find patterns.",
        "Climate change poses significant challenges for future generations.",
    ]

    # Initialize benchmark
    # benchmark = GREMLINBenchmark(model, tokenizer, device=device)

    # Run benchmark
    # results = benchmark.run_full_benchmark(
    #     test_prompts=test_prompts,
    #     output_path="F:/dev/GREMLIN_Claude_Code_Web_track/benchmark_results.json"
    # )

    print("\n✅ Benchmark pipeline ready!")
    print("   (Uncomment model loading to run full benchmark)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
