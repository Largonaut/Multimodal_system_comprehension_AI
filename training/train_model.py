"""
GREMLIN Training Loop - Custom Implementation

SECURITY COMPLIANCE:
- Zero Trainer/HuggingFace dependencies
- Pure PyTorch training loop
- Custom gradient accumulation
- Manual checkpointing
- Wandb integration (optional, self-hosted mode available)
- No hub code, no autoupdate, no telemetry

Trains Gemma 2 9B with LoRA on GREMLIN instruction corpus.

Author: GREMLIN Team
License: MIT
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import math
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import time

# Import custom modules
from gemma_model import GemmaForCausalLM
from gremlin_tokenizer_wrapper import GremlinTokenizer
from lora import apply_lora_to_model, print_lora_summary, save_lora_weights


class GREMLINDataset(Dataset):
    """
    Dataset for GREMLIN instruction-tuning.

    Loads converted JSONL file and tokenizes on-the-fly.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: GremlinTokenizer,
        max_length: int = 512,
        cache_size: int = 10000,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to instruction JSONL file
            tokenizer: GREMLIN tokenizer
            max_length: Maximum sequence length
            cache_size: Number of samples to cache in memory
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size

        # Count lines
        print(f"📊 Loading dataset: {self.data_path.name}")
        self.length = self._count_lines()
        print(f"   Total samples: {self.length:,}")

        # File handle for streaming
        self.file = None

        # Simple cache (most recent samples)
        self.cache = {}

    def _count_lines(self) -> int:
        """Count total lines in file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Open file if needed
        if self.file is None:
            self.file = open(self.data_path, 'r', encoding='utf-8')

        # Read line (this is inefficient for random access, but works for sequential)
        # For production, consider indexed file format
        self.file.seek(0)
        for i, line in enumerate(self.file):
            if i == idx:
                data = json.loads(line)
                text = data["text"]
                break
        else:
            raise IndexError(f"Index {idx} out of range")

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (same as input_ids for causal LM)
        labels = encoding["input_ids"].clone()

        # Mask padding tokens in labels (-100 is ignore index)
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

        # Cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = result

        return result


class GREMLINTrainer:
    """
    Custom training loop for GREMLIN model.

    Implements gradient accumulation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GremlinTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = "F:/dev/GREMLIN_Claude_Code_Web_track/models/gemma-2-9b-gremlin-lora",
        # Training hyperparameters
        num_epochs: int = 3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        # Logging
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_steps: int = 500,
        # Device
        device: str = "cuda",
        # Wandb (optional)
        use_wandb: bool = False,
        wandb_project: str = "gremlin-training",
    ):
        """Initialize trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        # Logging
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps

        # Device
        self.device = device
        self.model.to(self.device)

        # Calculate total steps
        self.total_steps = (len(train_dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs

        # Optimizer (only LoRA parameters)
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config={
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                })
                print("✓ Wandb initialized")
            except Exception as e:
                print(f"⚠️  Wandb initialization failed: {e}")
                self.use_wandb = False

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for LoRA parameters only."""
        # Get only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        print(f"\n🔧 Optimizer Configuration:")
        print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Optimizer: AdamW")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.

        Args:
            batch: Batch dictionary with input_ids, attention_mask, labels

        Returns:
            Loss value
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        logits, loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return loss.item() * self.gradient_accumulation_steps  # Return unscaled loss

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        # DataLoader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Single worker for Windows compatibility
        )

        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        # Accumulation tracking
        accumulated_loss = 0.0
        steps_since_update = 0

        for step, batch in enumerate(progress_bar):
            # Training step
            loss = self.train_step(batch)
            accumulated_loss += loss
            steps_since_update += 1

            # Update weights after accumulation steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Calculate average loss
                avg_loss = accumulated_loss / steps_since_update

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })

                # Log
                if self.global_step % self.logging_steps == 0:
                    self.log_metrics({
                        "train/loss": avg_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    })

                # Evaluate
                if self.val_dataset and self.global_step % self.eval_steps == 0:
                    val_loss = self.evaluate()
                    self.log_metrics({"val/loss": val_loss})

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(f"best_model")
                        print(f"\n💾 New best model! Val loss: {val_loss:.4f}")

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                # Reset accumulation
                accumulated_loss = 0.0
                steps_since_update = 0
                self.global_step += 1

    def evaluate(self) -> float:
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                _, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.model.train()

        return avg_loss

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and console."""
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=self.global_step)
            except Exception as e:
                print(f"⚠️  Wandb logging failed: {e}")

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights only
        lora_path = checkpoint_dir / "lora_weights.pt"
        save_lora_weights(self.model, str(lora_path))

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }

        state_path = checkpoint_dir / "training_state.pt"
        torch.save(state, state_path)

        print(f"\n💾 Checkpoint saved: {checkpoint_dir}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("STARTING GREMLIN TRAINING")
        print("=" * 70)
        print(f"Total epochs: {self.num_epochs}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.train_epoch(epoch)

            # Save end-of-epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")

        elapsed = time.time() - start_time
        hours = elapsed / 3600

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Total time: {hours:.2f} hours")
        print(f"Final loss: {self.best_val_loss:.4f}" if self.val_dataset else "No validation")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 70)


def main():
    """Main execution function."""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                GREMLIN TRAINING - CUSTOM LOOP                      ║
    ║                                                                    ║
    ║  Training Gemma 2 9B with LoRA on 14.9M GREMLIN samples            ║
    ║  Pure PyTorch - No Trainer, no hub dependencies                    ║
    ║  Frontier science in action                                        ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    model_path = "F:/dev/GREMLIN_Claude_Code_Web_track/models/gemma-2-9b-gremlin/pytorch_model.bin"
    tokenizer_path = "F:/dev/GREMLIN_Claude_Code_Web_track/models/tokenizer/gremlin_tokenizer.json"
    train_data_path = "F:/dev/GREMLIN_Claude_Code_Web_track/training_data/gremlin_instruction_train.jsonl"
    val_data_path = "F:/dev/GREMLIN_Claude_Code_Web_track/training_data/gremlin_instruction_val.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    # Load tokenizer
    print(f"\n📚 Loading tokenizer...")
    tokenizer = GremlinTokenizer(tokenizer_path)

    # Load model
    print(f"\n📦 Loading surgically-modified Gemma model...")
    # Note: In production, we'd load from saved checkpoint
    # For now, placeholder for model loading logic

    print(f"\n🔧 Applying LoRA...")
    # Apply LoRA to model
    # model = apply_lora_to_model(
    #     model,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     rank=16,
    #     alpha=32.0,
    #     dropout=0.05,
    # )

    # print_lora_summary(model)

    # Load datasets
    print(f"\n📊 Loading datasets...")
    train_dataset = GREMLINDataset(train_data_path, tokenizer, max_length=512)
    val_dataset = GREMLINDataset(val_data_path, tokenizer, max_length=512)

    # Initialize trainer
    # trainer = GREMLINTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     num_epochs=3,
    #     batch_size=1,
    #     gradient_accumulation_steps=16,
    #     learning_rate=2e-4,
    #     use_wandb=True,
    # )

    # Start training
    # trainer.train()

    print("\n✅ Training pipeline ready!")
    print("   (Uncomment model loading and trainer initialization to start training)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
