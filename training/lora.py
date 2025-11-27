"""
Low-Rank Adaptation (LoRA) - Custom Implementation

SECURITY COMPLIANCE:
- Zero PEFT/Hugging Face dependencies
- Pure PyTorch implementation
- No hub code, no autoupdate, no telemetry
- Built from first principles

LoRA decomposes weight updates into low-rank matrices:
    W' = W + BA
where:
- W: Original frozen weights (d_in × d_out)
- B: Low-rank matrix (d_in × r)
- A: Low-rank matrix (r × d_out)
- r: Rank (typically 4-16, much smaller than d_in or d_out)

This reduces trainable parameters from (d_in × d_out) to (d_in × r + r × d_out).

Author: GREMLIN Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math


class LoRALayer(nn.Module):
    """
    LoRA layer wrapper for a linear layer.

    Adds low-rank adaptation matrices A and B to an existing frozen linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (bottleneck dimension)
            alpha: LoRA scaling factor (typically 2×rank)
            dropout: Dropout probability
        """
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        # A: (rank × out_features) - initialized with random Gaussian
        # B: (in_features × rank) - initialized to zero
        self.lora_A = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_B = nn.Parameter(torch.zeros(in_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Initialize A with Kaiming uniform (good for ReLU-like activations)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B starts at zero, so initial LoRA contribution is zero (safe initialization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA output: x @ B @ A * scaling
        """
        # Apply dropout to input
        x_dropped = self.dropout(x)

        # Low-rank transformation: x @ B @ A
        # Step 1: x @ B  (project to rank-dimensional space)
        result = torch.matmul(x_dropped, self.lora_B)

        # Step 2: result @ A  (project back to output space)
        result = torch.matmul(result, self.lora_A)

        # Scale by alpha/rank
        result = result * self.scaling

        return result


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Combines frozen base linear layer with trainable LoRA matrices.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        """
        Initialize LoRA-augmented linear layer.

        Args:
            base_layer: Original linear layer (will be frozen)
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout probability
        """
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Freeze base layer
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)

        # LoRA adaptation
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base output + LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            base_layer(x) + lora(x)
        """
        # Base transformation (frozen)
        base_output = self.base_layer(x)

        # LoRA adaptation (trainable)
        lora_output = self.lora(x)

        # Combine
        return base_output + lora_output

    def merge_weights(self):
        """
        Merge LoRA weights into base layer.

        After training, we can merge A and B into the base weights:
            W' = W + B @ A * scaling

        This eliminates inference overhead.
        """
        with torch.no_grad():
            # Compute LoRA weight update: B @ A * scaling
            lora_weight = torch.matmul(self.lora.lora_B, self.lora.lora_A) * self.lora.scaling

            # Add to base weights
            self.base_layer.weight.data += lora_weight.T

            # Zero out LoRA matrices (they're now merged)
            self.lora.lora_A.data.zero_()
            self.lora.lora_B.data.zero_()


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.

    Args:
        model: Model to augment with LoRA
        target_modules: Names of modules to apply LoRA to (e.g., ["q_proj", "v_proj"])
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability

    Returns:
        Model with LoRA layers injected
    """
    print(f"🔧 Applying LoRA to model...")
    print(f"   Rank: {rank}")
    print(f"   Alpha: {alpha}")
    print(f"   Dropout: {dropout}")
    print(f"   Target modules: {target_modules}")

    lora_layers_added = 0

    # Recursively find and replace target modules
    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        module_name = name.split('.')[-1]  # Get last part of name (e.g., "q_proj")

        if module_name in target_modules and isinstance(module, nn.Linear):
            # Get parent module
            parent = model
            parts = name.split('.')[:-1]
            for part in parts:
                parent = getattr(parent, part)

            # Replace with LoRA version
            lora_linear = LoRALinear(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

            setattr(parent, module_name, lora_linear)
            lora_layers_added += 1

            print(f"   ✓ Added LoRA to {name} ({module.in_features} → {module.out_features})")

    print(f"\n✓ LoRA applied to {lora_layers_added} layers")

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable vs. total parameters in model.

    Returns:
        Dictionary with 'trainable', 'frozen', and 'total' counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
    }


def print_lora_summary(model: nn.Module):
    """Print summary of LoRA configuration."""
    print("\n" + "=" * 70)
    print("LORA CONFIGURATION SUMMARY")
    print("=" * 70)

    # Count parameters
    param_counts = count_parameters(model)

    print(f"\nParameter Counts:")
    print(f"   Total:      {param_counts['total']:>12,} parameters")
    print(f"   Trainable:  {param_counts['trainable']:>12,} parameters")
    print(f"   Frozen:     {param_counts['frozen']:>12,} parameters")
    print(f"   Ratio:      {param_counts['trainable'] / param_counts['total'] * 100:>11.2f}% trainable")

    # Find LoRA layers
    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_layers.append((name, module))

    print(f"\nLoRA Layers: {len(lora_layers)}")
    for name, module in lora_layers:
        lora_params = module.lora.lora_A.numel() + module.lora.lora_B.numel()
        base_params = module.base_layer.weight.numel()
        ratio = (lora_params / base_params) * 100

        print(f"   {name}")
        print(f"      Shape: {module.in_features} → {module.out_features}")
        print(f"      Rank:  {module.lora.rank}")
        print(f"      LoRA params: {lora_params:,} ({ratio:.2f}% of base)")

    print("=" * 70)


def merge_all_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base layers.

    After training, this creates a single unified model with no LoRA overhead.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights
    """
    print("\n🔀 Merging LoRA weights into base model...")

    merged_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged_count += 1
            print(f"   ✓ Merged {name}")

    print(f"\n✓ Merged {merged_count} LoRA layers")

    return model


def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save only LoRA weights (not full model).

    This is much more space-efficient than saving the entire model.

    Args:
        model: Model with LoRA layers
        save_path: Path to save LoRA weights
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B.data

    torch.save(lora_state_dict, save_path)
    print(f"💾 Saved LoRA weights to {save_path}")
    print(f"   Size: {len(lora_state_dict)} tensors")


def load_lora_weights(model: nn.Module, load_path: str):
    """
    Load LoRA weights from file.

    Args:
        model: Model with LoRA layers (must match saved structure)
        load_path: Path to LoRA weights file
    """
    lora_state_dict = torch.load(load_path)

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"

            if lora_a_key in lora_state_dict:
                module.lora.lora_A.data = lora_state_dict[lora_a_key]
            if lora_b_key in lora_state_dict:
                module.lora.lora_B.data = lora_state_dict[lora_b_key]

    print(f"📥 Loaded LoRA weights from {load_path}")


if __name__ == "__main__":
    # Test LoRA implementation
    print("Testing LoRA implementation...")
    print("=" * 70)

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(512, 512)
            self.k_proj = nn.Linear(512, 512)
            self.v_proj = nn.Linear(512, 512)
            self.o_proj = nn.Linear(512, 512)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            return self.o_proj(v)

    model = SimpleModel()

    print("\n📊 Original Model:")
    original_params = count_parameters(model)
    print(f"   Total parameters: {original_params['total']:,}")

    # Apply LoRA
    model = apply_lora_to_model(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        rank=16,
        alpha=32.0,
        dropout=0.05,
    )

    # Print summary
    print_lora_summary(model)

    # Test forward pass
    print("\n✅ Testing forward pass...")
    x = torch.randn(2, 10, 512)  # Batch of 2, sequence length 10, hidden dim 512
    output = model(x)
    print(f"   Input shape:  {list(x.shape)}")
    print(f"   Output shape: {list(output.shape)}")
    print(f"   ✓ Forward pass successful")

    # Test merging
    print("\n🔀 Testing weight merging...")
    model = merge_all_lora_weights(model)
    print("   ✓ Weights merged successfully")

    print("\n" + "=" * 70)
    print("✅ LoRA IMPLEMENTATION TEST COMPLETE")
    print("=" * 70)
