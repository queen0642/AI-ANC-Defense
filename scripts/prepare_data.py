"""
prepare_data.py
----------------
Builds a noisy/clean training set for the AI-ANC-Defense denoiser by mixing:
  - Clean speech        -> VCTK corpus  (data/clean/*.wav)
  - Military noise      -> MAD (Military Audio Dataset) (data/noise_military/*.wav)
  - General/urban noise -> UrbanSound8K  (data/noise_urban/*.wav)

Folder layout expected BEFORE running this script (see README for how to get here):

data/
  clean/            <- .wav files from VCTK (any speakers, 16kHz mono preferred)
  noise_military/   <- .wav files from the MAD Kaggle dataset (gunshots, helicopters,
                        artillery, vehicles, etc. - keep the folder flat, one .wav per clip)
  noise_urban/      <- .wav files from UrbanSound8K (used as a secondary / distractor
                        noise class so the model doesn't overfit only to military noise)

Output (created by this script):

data/processed/
  train/noisy/*.wav
  train/clean/*.wav
  val/noisy/*.wav
  val/clean/*.wav

Usage:
    python scripts/prepare_data.py --n_train 1500 --n_val 200
"""
import argparse
import random
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

SR = 16000


def load_wav(path):
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav


def fit_noise_to_length(noise, length):
    if len(noise) < length:
        reps = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, reps)
    start = random.randint(0, max(0, len(noise) - length))
    return noise[start:start + length]


def mix_at_snr(speech, noise, snr_db):
    speech_power = np.mean(speech ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(speech_power / (noise_power * (10 ** (snr_db / 10))))
    noisy = speech + noise * scale
    # avoid clipping
    peak = np.max(np.abs(noisy)) + 1e-8
    if peak > 0.99:
        noisy = noisy / peak * 0.99
        speech = speech / peak * 0.99
    return noisy, speech


def build_split(clean_files, military_files, urban_files, out_dir, n_pairs,
                 snr_range=(-5, 15), military_ratio=0.7, min_len_s=2.0, max_len_s=6.0):
    noisy_dir = out_dir / "noisy"
    clean_dir = out_dir / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_pairs), desc=f"Building {out_dir.name}"):
        speech_path = random.choice(clean_files)
        speech = load_wav(speech_path)

        target_len = int(random.uniform(min_len_s, max_len_s) * SR)
        if len(speech) < target_len:
            speech = np.pad(speech, (0, target_len - len(speech)))
        else:
            start = random.randint(0, len(speech) - target_len)
            speech = speech[start:start + target_len]

        # Mostly train on military noise (this is the point of the project),
        # but mix in some urban noise so the model stays general and robust.
        use_military = random.random() < military_ratio
        noise_pool = military_files if use_military else urban_files
        noise_path = random.choice(noise_pool)
        noise = load_wav(noise_path)
        noise = fit_noise_to_length(noise, target_len)

        snr_db = random.uniform(*snr_range)
        noisy, clean = mix_at_snr(speech, noise, snr_db)

        fname = f"{i:05d}.wav"
        sf.write(noisy_dir / fname, noisy, SR)
        sf.write(clean_dir / fname, clean, SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--military_ratio", type=float, default=0.7,
                     help="Fraction of mixes that use military noise vs. urban noise")
    args = ap.parse_args()

    root = Path(args.data_root)
    clean_files = sorted((root / "clean").glob("*.wav"))
    military_files = sorted((root / "noise_military").glob("*.wav"))
    urban_files = sorted((root / "noise_urban").glob("*.wav"))

    assert clean_files, f"No clean speech .wav files found in {root/'clean'}"
    assert military_files, f"No military noise .wav files found in {root/'noise_military'}"
    assert urban_files, f"No urban noise .wav files found in {root/'noise_urban'}"

    print(f"Found {len(clean_files)} clean, {len(military_files)} military noise, "
          f"{len(urban_files)} urban noise files.")

    processed = root / "processed"
    build_split(clean_files, military_files, urban_files, processed / "train",
                args.n_train, military_ratio=args.military_ratio)
    build_split(clean_files, military_files, urban_files, processed / "val",
                args.n_val, military_ratio=args.military_ratio)

    print("Done. Training pairs in data/processed/train, validation pairs in data/processed/val")


if __name__ == "__main__":
    main()
