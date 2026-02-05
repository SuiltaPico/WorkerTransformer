"""
WorkerTransformer Benchmark Script

Compares:
1. Standard Transformer (Baseline)
2. WorkerTransformer (Ours)

Goal:
Verify speedup and convergence on a small-scale character-level dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from baseline import StdTransformer
from model import InplaceWorkerTransformer


class SimpleTextDataset(Dataset):
    """Simple Character-level Dataset"""
    def __init__(self, text: str, block_size: int, vocab=None):
        self.block_size = block_size
        
        if vocab is None:
            # Build vocabulary from text
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        else:
            self.char_to_idx, self.idx_to_char = vocab
            
        self.vocab_size = len(self.char_to_idx)
        
        # Encode text, handling unknown characters if any
        self.data = torch.tensor([self.char_to_idx.get(ch, 0) for ch in text], dtype=torch.long)
        print(f"Dataset Size: {len(self.data)} tokens, Vocab Size: {self.vocab_size}")
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    
    def decode(self, indices):
        """Decode indices to string"""
        return ''.join([self.idx_to_char[i.item()] for i in indices])


def train_model(model, train_loader, val_loader, max_steps, device, model_name="Model"):
    """Train a single model for a fixed number of steps"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    model.train()
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    total_loss = 0
    num_batches = 0
    interval_loss = 0
    interval_batches = 0
    step = 0
    start_time = time.time()
    
    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break
                
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            if loss is None or not torch.isfinite(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            interval_loss += loss.item()
            interval_batches += 1
            step += 1
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                interval_avg = interval_loss / max(interval_batches, 1)
                
                # Lightweight Validation
                model.eval()
                val_loss_accum = 0.0
                val_steps = 0
                max_val_batches = 20  # Check 20 batches
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        _, vloss = model(vx, vy)
                        val_loss_accum += vloss.item()
                        val_steps += 1
                        if val_steps >= max_val_batches:
                            break
                val_loss_cur = val_loss_accum / max(val_steps, 1)
                model.train()
                
                print(f"Step {step}/{max_steps}, Train Loss: {interval_avg:.4f}, Val Loss: {val_loss_cur:.4f}, Speed: {steps_per_sec:.2f} steps/s")
                
                # Reset interval stats
                interval_loss = 0
                interval_batches = 0
    
    total_time = time.time() - start_time
    final_train_loss = total_loss / max(num_batches, 1)
    avg_speed = max_steps / total_time
    
    # Validation Loop
    print(f"Running Validation...")
    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            val_loss += loss.item()
            val_batches += 1
            if val_batches >= 50: # Limit validation batches for speed
                break
    
    final_val_loss = val_loss / max(val_batches, 1)
    
    print(f"Training Finished!")
    print(f"Final Train Loss: {final_train_loss:.4f}")
    print(f"Final Val Loss:   {final_val_loss:.4f}")
    print(f"Avg Speed: {avg_speed:.2f} steps/s")
    
    return final_train_loss, final_val_loss, avg_speed


def main():
    # Hyperparameters
    batch_size = 16
    block_size = 1024  # Long sequence to demonstrate sparse attention benefits
    dim = 256
    num_heads = 4
    num_layers = 4
    worker_interval = 4
    max_steps = 2000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Sequence Length: {block_size}")
    print(f"Worker Interval: {worker_interval}")
    print(f"Model Dim: {dim}, Layers: {num_layers}")
    
    # Load Data
    data_path = Path(__file__).parent / "input.txt"
    if not data_path.exists():
        print("Warning: input.txt not found. Using synthetic data.")
        text = "Standard Transformer vs Inplace Worker Transformer. " * 10000
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    # Split into train/val
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Build shared vocab
    chars = sorted(list(set(text)))
    vocab = ({ch: i for i, ch in enumerate(chars)}, {i: ch for i, ch in enumerate(chars)})
    
    train_dataset = SimpleTextDataset(train_text, block_size, vocab=vocab)
    val_dataset = SimpleTextDataset(val_text, block_size, vocab=vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ============ Model 1: Standard Transformer ============
    print("\n" + "="*70)
    print("Model 1: Standard Transformer (Baseline)")
    print("="*70)
    
    model1 = StdTransformer(
        vocab_size=train_dataset.vocab_size,
        block_size=block_size,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        ffn_hidden_dim=dim * 4,
    ).to(device)
    
    num_params1 = sum(p.numel() for p in model1.parameters())
    print(f"Params: {num_params1:,}")
    
    train_loss1, val_loss1, speed1 = train_model(model1, train_loader, val_loader, max_steps, device, "StdTransformer")
    
    # Clean up memory
    del model1
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    # ============ Model 2: WorkerTransformer ============
    print("\n" + "="*70)
    print("Model 2: WorkerTransformer (Challenger)")
    print("="*70)
    
    model2 = InplaceWorkerTransformer(
        vocab_size=train_dataset.vocab_size,
        block_size=block_size,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        ffn_hidden_dim=dim * 4,
        worker_interval=worker_interval,
    ).to(device)
    
    num_params2 = sum(p.numel() for p in model2.parameters())
    print(f"Params: {num_params2:,}")
    
    train_loss2, val_loss2, speed2 = train_model(model2, train_loader, val_loader, max_steps, device, "WorkerTransformer")

    # ============ Comparison Report ============
    print("\n" + "="*70)
    print("Final Comparison: Standard vs WorkerTransformer")
    print("="*70)
    print(f"{'Model':<20} | {'Params':<10} | {'Train Loss':<10} | {'Val Loss':<10} | {'Speed (steps/s)':<15} | {'Speedup':<10}")
    print("-" * 85)
    print(f"{'Standard':<20} | {num_params1:<10,} | {train_loss1:.4f}     | {val_loss1:.4f}   | {speed1:.2f}            | 1.00x")
    print(f"{'WorkerTransformer':<20} | {num_params2:<10,} | {train_loss2:.4f}     | {val_loss2:.4f}   | {speed2:.2f}            | {speed2/speed1:.2f}x")
    print("-" * 85)
    
    print("\nWorkerTransformer Generation Test:")
    model2.eval()
    seed_text = text[:20]
    seed_idx = torch.tensor(
        [train_dataset.char_to_idx[ch] for ch in seed_text], 
        dtype=torch.long
    ).unsqueeze(0).to(device)
    generated = model2.generate(seed_idx, max_new_tokens=50)
    print(f"Seed: {seed_text}")
    print(f"Generated: {train_dataset.decode(generated[0])[len(seed_text):]}")

if __name__ == "__main__":
    main()
