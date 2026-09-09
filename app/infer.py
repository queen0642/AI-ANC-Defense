"""
infer.py
--------
Loads the (optionally fine-tuned) denoiser model once and exposes a single
function, denoise_file(), that the Streamlit app (and any CLI script) calls.
"""
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa
from denoiser import pretrained
from denoiser.dsp import convert_audio

SR = 16000
FINETUNED_PATH = Path(__file__).resolve().parent.parent / "models" / "finetuned_military.pth"

_model_cache = {}


def load_model(use_finetuned: bool = True):
    """Loads the pretrained dns64 model, applying the fine-tuned weights if available."""
    key = "finetuned" if (use_finetuned and FINETUNED_PATH.exists()) else "pretrained"
    if key in _model_cache:
        return _model_cache[key]

    model = pretrained.dns64()
    if key == "finetuned":
        state = torch.load(FINETUNED_PATH, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    _model_cache[key] = model
    return model


def denoise_array(wav: np.ndarray, sr: int, use_finetuned: bool = True) -> np.ndarray:
    """Takes a mono float32 waveform at any sample rate, returns the enhanced waveform at 16kHz."""
    model = load_model(use_finetuned)

    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)

    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    wav_tensor = convert_audio(wav_tensor, SR, model.sample_rate, model.chin)

    with torch.no_grad():
        enhanced = model(wav_tensor)[0]

    return enhanced.numpy().squeeze()


def denoise_file(in_path: str, out_path: str, use_finetuned: bool = True):
    """Reads a wav/mp3 file, denoises it, writes the result. Returns (clean_path, sr)."""
    wav, sr = librosa.load(in_path, sr=None, mono=True)
    enhanced = denoise_array(wav, sr, use_finetuned=use_finetuned)
    sf.write(out_path, enhanced, SR)
    return out_path, SR
