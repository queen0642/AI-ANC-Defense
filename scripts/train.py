"""
train.py
--------
Fine-tunes Facebook's pretrained Denoiser (dns64) on the military-noise mixes
built by prepare_data.py. This is FINE-TUNING, not training from scratch --
we start from a checkpoint already trained on a large general noise corpus,
and nudge it toward gunshots / helicopters / artillery / vehicles specifically.

Usage:
    python scripts/train.py --epochs 5 --batch_size 4 --lr 1e-5 \
        --data_dir data/processed --out models/finetuned_military.pth
"""
import argparse
from pathlib import Path

import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from denoiser import pretrained

SR = 16000


class NoisyCleanDataset(Dataset):
    def __init__(self, split_dir, segment_len=SR * 4):
        split_dir = Path(split_dir)
        self.noisy_files = sorted((split_dir / "noisy").glob("*.wav"))
        self.clean_files = sorted((split_dir / "clean").glob("*.wav"))
        assert len(self.noisy_files) == len(self.clean_files)
        self.segment_len = segment_len

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy, _ = sf.read(self.noisy_files[idx], dtype="float32")
        clean, _ = sf.read(self.clean_files[idx], dtype="float32")
        # pad/crop to a fixed segment length so batches stack cleanly
        n = self.segment_len
        if len(noisy) < n:
            pad = n - len(noisy)
            noisy = torch.nn.functional.pad(torch.tensor(noisy), (0, pad)).numpy()
            clean = torch.nn.functional.pad(torch.tensor(clean), (0, pad)).numpy()
        else:
            noisy = noisy[:n]
            clean = clean[:n]
        return torch.tensor(noisy, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32)


def si_snr_loss(estimate, target, eps=1e-8):
    """Scale-Invariant Signal-to-Noise Ratio loss (negative SI-SNR, minimised)."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    s_target = (torch.sum(estimate * target, dim=-1, keepdim=True) * target) / \
               (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    e_noise = estimate - s_target
    si_snr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    return -si_snr.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="models/finetuned_military.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = pretrained.dns64().to(device)
    model.train()

    train_ds = NoisyCleanDataset(Path(args.data_dir) / "train")
    val_ds = NoisyCleanDataset(Path(args.data_dir) / "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.unsqueeze(1).to(device), clean.unsqueeze(1).to(device)
            estimate = model(noisy)
            loss = si_snr_loss(estimate.squeeze(1), clean.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.unsqueeze(1).to(device), clean.unsqueeze(1).to(device)
                estimate = model(noisy)
                val_loss += si_snr_loss(estimate.squeeze(1), clean.squeeze(1)).item()
        avg_val = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch+1}/{args.epochs} | train SI-SNR loss: {avg_train:.3f} "
              f"| val SI-SNR loss: {avg_val:.3f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), args.out)
            print(f"  -> New best model saved to {args.out}")

    print("Training complete. Best validation loss:", best_val)


if __name__ == "__main__":
    main()
