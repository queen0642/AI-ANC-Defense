# Adaptive Noise Cancellation

This project is a prototype for AI/ML-enabled adaptive noise cancellation,
focused on suppressing defence-related noises while preserving speech
intelligibility.

## Folder structure

- `data/clean/` - Clean speech and reference audio without added noise.
- `data/noise/` - Noise recordings such as rotor, engine, drone, siren,
  gunshot, and artillery sounds.
- `data/noisy/` - Speech mixed with noise for training or development.
- `data/test/` - Held-out audio used for testing and evaluation.
- `models/` - Trained model files and model-related artifacts.
- `src/preprocessing/` - Audio loading, cleaning, feature extraction, and
  preparation utilities.
- `src/inference/` - Code for applying noise cancellation to new audio.
- `src/evaluation/` - Metrics and evaluation workflows for suppression and
  speech intelligibility.
- `src/adaptive_filter/` - Adaptive filtering algorithms and supporting
  components.
- `results/` - Experiment outputs, metrics, and generated audio results.
- `demo/` - Demonstrations and prototype application assets.
# AI/ML-Enabled Adaptive Noise Cancellation

An AI-driven adaptive noise cancellation system designed to suppress
stationary, non-stationary, and impulsive noise while preserving speech
intelligibility for real-time communication.

## Project Objective

The system combines AI-based speech enhancement with adaptive filtering
to reduce complex environmental noise.

## Pipeline

Noisy Speech
    ↓
AI Noise Suppression
    ↓
Residual Noise Reduction
    ↓
Adaptive NLMS Filter
    ↓
Enhanced Speech
    ↓
Headphones / Communication System

## Technologies

- Python
- PyTorch
- Torchaudio
- Librosa
- SoundDevice
- ONNX Runtime
- Raspberry Pi 4B

## Evaluation

The system will be evaluated using:

- SNR
- STOI
- PESQ

## Status

🚧 Project under development.