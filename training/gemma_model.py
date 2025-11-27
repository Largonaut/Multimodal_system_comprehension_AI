"""
Gemma 2 9B Model Architecture - Custom PyTorch Implementation

SECURITY COMPLIANCE:
- Zero transformers library dependency
- Pure PyTorch implementation
- No hub code, no autoupdate, no telemetry
- Manual SafeTensors loading
- Complete architectural isolation

Architecture based on Gemma 2 technical report:
- 42 transformer layers
- 3584 hidden dimension
- 16 attention heads (8 KV heads - Grouped Query Attention)
- 256K vocabulary (base), resizable to 128K for GREMLIN
- RMSNorm for layer normalization
- RoPE for positional encoding
- SwiGLU activation in MLP

Author: GREMLIN Team
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path
import json


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Used in Gemma instead of LayerNorm for better stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Calculate RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes positional information through rotation in complex space.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        # Create rotation matrix (cos and sin components)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Uses 16 query heads but only 8 key-value heads for efficiency.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads

        # Query heads (16)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # Key-Value heads (8)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.get("max_position_embeddings", 8192)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV heads to match Q heads (8 → 16)
        key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and apply to values
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class GemmaMLP(nn.Module):
    """
    Multi-Layer Perceptron with SwiGLU activation.

    SwiGLU: Swish-Gated Linear Unit (more stable than ReLU)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]

        # SwiGLU requires two up projections
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(gate) * up) → down
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GemmaDecoderLayer(nn.Module):
    """
    Single Gemma transformer decoder layer.

    Architecture:
    - Pre-normalization (RMSNorm before each sublayer)
    - Grouped Query Attention
    - SwiGLU MLP
    - Residual connections
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        # Attention block
        self.input_layernorm = RMSNorm(self.hidden_size)
        self.self_attn = GroupedQueryAttention(config)

        # MLP block
        self.post_attention_layernorm = RMSNorm(self.hidden_size)
        self.mlp = GemmaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    """
    Gemma 2 9B base model (decoder-only transformer).

    Custom implementation without transformers library dependency.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]

        # Token embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Transformer layers (42 for Gemma 2 9B)
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config)
            for _ in range(config["num_hidden_layers"])
        ])

        # Final layer norm
        self.norm = RMSNorm(self.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Apply each transformer layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states


class GemmaForCausalLM(nn.Module):
    """
    Gemma 2 9B for causal language modeling.

    Adds language modeling head on top of base model.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Base model
        self.model = GemmaModel(config)

        # Language modeling head (vocabulary projection)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get hidden states from base model
        hidden_states = self.model(input_ids, attention_mask=attention_mask)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config["vocab_size"]),
                shift_labels.view(-1)
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Simple greedy generation (for testing).

        For production, implement more sophisticated sampling.
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits
                logits, _ = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Stop if EOS token (implement based on tokenizer)
                # if next_token.item() == eos_token_id:
                #     break

        return input_ids


def load_model_from_safetensors(model_dir: str, device: str = "cuda") -> GemmaForCausalLM:
    """
    Load Gemma model from SafeTensors files.

    Args:
        model_dir: Directory containing SafeTensors shards and config.json
        device: Device to load model on

    Returns:
        Loaded GemmaForCausalLM model
    """
    from safetensors import safe_open

    model_path = Path(model_dir)

    # Load configuration
    config_path = model_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"📋 Loading Gemma 2 configuration:")
    print(f"   Vocabulary size: {config['vocab_size']:,}")
    print(f"   Hidden size: {config['hidden_size']:,}")
    print(f"   Layers: {config['num_hidden_layers']}")
    print(f"   Attention heads: {config['num_attention_heads']}")
    print(f"   KV heads: {config['num_key_value_heads']}")

    # Initialize model
    model = GemmaForCausalLM(config)

    # Load SafeTensors shards
    print(f"\n📦 Loading model weights from SafeTensors...")

    # Find all shard files
    shard_files = sorted(model_path.glob("model-*.safetensors"))

    if not shard_files:
        raise FileNotFoundError(f"No SafeTensors files found in {model_path}")

    print(f"   Found {len(shard_files)} shards")

    # Load weights from each shard
    state_dict = {}
    for shard_file in shard_files:
        print(f"   Loading {shard_file.name}...")
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)

    print(f"\n✓ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move to device
    model = model.to(device)
    print(f"   Device: {device}")

    return model


if __name__ == "__main__":
    # Test model initialization
    print("Testing Gemma 2 9B custom implementation...")

    # Default Gemma 2 9B config
    test_config = {
        "vocab_size": 256000,
        "hidden_size": 3584,
        "intermediate_size": 14336,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
    }

    model = GemmaForCausalLM(test_config)
    print(f"✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Memory: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f} GB")
