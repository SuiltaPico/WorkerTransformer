"""
WorkerTransformer Time-Based Benchmark Script
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
    def __init__(self, text: str, block_size: int, vocab=None):
        self.block_size = block_size
        
        if vocab is None:
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        else:
            self.char_to_idx, self.idx_to_char = vocab
            
        self.vocab_size = len(self.char_to_idx)
        self.data = torch.tensor([self.char_to_idx.get(ch, 0) for ch in text], dtype=torch.long)
        print(f"Dataset Size: {len(self.data)} tokens, Vocab Size: {self.vocab_size}")
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i.item()] for i in indices])


def train_model(model, train_loader, val_loader, max_steps, device, model_name="Model", time_limit_sec=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    model.train()
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    if time_limit_sec:
        print(f"Time Limit: {time_limit_sec} seconds")
    
    total_loss = 0
    num_batches = 0
    interval_loss = 0
    interval_batches = 0
    step = 0
    start_time = time.time()
    
    train_loss_history = []
    val_loss_history = []
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
                
    train_iterator = cycle(train_loader)
    
    while True:
        if time_limit_sec and (time.time() - start_time) > time_limit_sec:
            print(f"Time limit reached ({time_limit_sec}s). Stopping.")
            break
        if not time_limit_sec and step >= max_steps:
            print(f"Max steps reached ({max_steps}). Stopping.")
            break
            
        x, y = next(train_iterator)
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
        
        elapsed = time.time() - start_time
        
        if step % 100 == 0:
            steps_per_sec = step / elapsed
            interval_avg = interval_loss / max(interval_batches, 1)
            
            model.eval()
            val_loss_accum = 0.0
            val_steps = 0
            max_val_batches = 20
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
            
            train_loss_history.append((elapsed, interval_avg))
            val_loss_history.append((elapsed, val_loss_cur))
            
            print(f"Time: {elapsed:.1f}s | Step {step}, Train Loss: {interval_avg:.4f}, Val Loss: {val_loss_cur:.4f}, Speed: {steps_per_sec:.2f} steps/s")
            
            interval_loss = 0
            interval_batches = 0
    
    total_time = time.time() - start_time
    final_train_loss = total_loss / max(num_batches, 1)
    avg_speed = step / total_time
    
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
            if val_batches >= 50:
                break
    
    final_val_loss = val_loss / max(val_batches, 1)
    
    print(f"Training Finished!")
    print(f"Final Train Loss: {final_train_loss:.4f}")
    print(f"Final Val Loss:   {final_val_loss:.4f}")
    print(f"Avg Speed: {avg_speed:.2f} steps/s")
    
    return final_train_loss, final_val_loss, avg_speed, val_loss_history


def main():
    batch_size = 16
    block_size = 1024
    dim = 256
    num_heads = 4
    num_layers = 4
    worker_interval = 4
    
    time_limit_sec = 300
    max_steps = 2000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Sequence Length: {block_size}")
    print(f"Worker Interval: {worker_interval}")
    print(f"Model Dim: {dim}, Layers: {num_layers}")
    
    data_path = Path(__file__).parent / "input.txt"
    if not data_path.exists():
        print("Warning: input.txt not found. Using synthetic data.")
        text = "Standard Transformer vs Inplace Worker Transformer. " * 10000
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    chars = sorted(list(set(text)))
    vocab = ({ch: i for i, ch in enumerate(chars)}, {i: ch for i, ch in enumerate(chars)})
    
    train_dataset = SimpleTextDataset(train_text, block_size, vocab=vocab)
    val_dataset = SimpleTextDataset(val_text, block_size, vocab=vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
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
    
    t_loss1, v_loss1, speed1, v_hist1 = train_model(
        model1, train_loader, val_loader, max_steps, device, "StdTransformer", time_limit_sec=time_limit_sec
    )
    
    del model1
    torch.cuda.empty_cache() if device == 'cuda' else None
    
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
    
    t_loss2, v_loss2, speed2, v_hist2 = train_model(
        model2, train_loader, val_loader, max_steps, device, "WorkerTransformer", time_limit_sec=time_limit_sec
    )

    print("\n" + "="*95)
    print(f"Final Comparison: Standard vs WorkerTransformer ({time_limit_sec if time_limit_sec else 'Steps'}s Limit)")
    print("="*95)
    print(f"{'Model':<20} | {'Params':<10} | {'Train Loss':<10} | {'Val Loss':<10} | {'Speed (steps/s)':<15} | {'Speedup':<10}")
    print("-" * 95)
    print(f"{'Standard':<20} | {num_params1:<10,} | {t_loss1:.4f}     | {v_loss1:.4f}   | {speed1:.2f}            | 1.00x")
    print(f"{'WorkerTransformer':<20} | {num_params2:<10,} | {t_loss2:.4f}     | {v_loss2:.4f}   | {speed2:.2f}            | {speed2/speed1:.2f}x")
    print("-" * 95)
    
    print("\nLoss at Time Checkpoints:")
    checkpoints = [60, 120, 180, 240, 300]
    print(f"{'Time (s)':<10} | {'Std Val Loss':<15} | {'Worker Val Loss':<15} | {'Diff':<10}")
    print("-" * 60)
    
    def get_loss_at_time(hist, t):
        last_loss = float('inf')
        for time_point, loss in hist:
            if time_point > t:
                break
            last_loss = loss
        return last_loss

    for t in checkpoints:
        if t > time_limit_sec: break
        l1 = get_loss_at_time(v_hist1, t)
        l2 = get_loss_at_time(v_hist2, t)
        diff = l1 - l2
        print(f"{t:<10} | {l1:<15.4f} | {l2:<15.4f} | {diff:+.4f}")
    
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
