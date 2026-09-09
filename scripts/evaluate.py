"""
evaluate.py
-----------
Runs the (fine-tuned) model over data/processed/val and prints average
SNR / STOI / PESQ improvement -- this is the evidence you put on your results slide.

Usage:
    python scripts/evaluate.py --data_dir data/processed/val --use_finetuned
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "app"))
from infer import denoise_array  # noqa: E402

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/val")
    ap.add_argument("--use_finetuned", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Only evaluate first N pairs (faster)")
    args = ap.parse_args()

    noisy_dir = Path(args.data_dir) / "noisy"
    clean_dir = Path(args.data_dir) / "clean"
    noisy_files = sorted(noisy_dir.glob("*.wav"))
    if args.limit:
        noisy_files = noisy_files[:args.limit]

    snrs_before, snrs_after, stois_before, stois_after, pesqs_before, pesqs_after = ([] for _ in range(6))

    for nf in tqdm(noisy_files, desc="Evaluating"):
        cf = clean_dir / nf.name
        noisy, sr = sf.read(nf)
        clean, _ = sf.read(cf)

        enhanced = denoise_array(noisy.astype(np.float32), sr, use_finetuned=args.use_finetuned)

        n = min(len(clean), len(enhanced), len(noisy))
        c, e, ny = clean[:n], enhanced[:n], noisy[:n]

        snrs_before.append(10 * np.log10(np.sum(c ** 2) / (np.sum((ny - c) ** 2) + 1e-8)))
        snrs_after.append(10 * np.log10(np.sum(c ** 2) / (np.sum((e - c) ** 2) + 1e-8)))
        stois_before.append(stoi(c, ny, SR, extended=False))
        stois_after.append(stoi(c, e, SR, extended=False))
        try:
            pesqs_before.append(pesq(SR, c, ny, "wb"))
            pesqs_after.append(pesq(SR, c, e, "wb"))
        except Exception:
            pass  # pesq occasionally fails on silent/very short segments

    print("\n===== Results over", len(noisy_files), "validation pairs =====")
    print(f"SNR   before: {np.mean(snrs_before):.2f} dB  ->  after: {np.mean(snrs_after):.2f} dB")
    print(f"STOI  before: {np.mean(stois_before):.3f}    ->  after: {np.mean(stois_after):.3f}")
    if pesqs_before:
        print(f"PESQ  before: {np.mean(pesqs_before):.2f}    ->  after: {np.mean(pesqs_after):.2f}")


if __name__ == "__main__":
    main()
