"""
app.py
------
Standalone Streamlit app for AI-ANC-Defense.

Run with:
    streamlit run app/app.py

What it does:
  1. Lets you upload a noisy audio file (.wav/.mp3) - e.g. speech recorded with
     gunshots / helicopter rotor / vehicle engine noise in the background.
  2. Runs it through the AI denoiser model (fine-tuned on military noise if
     models/finetuned_military.pth exists, otherwise the general pretrained model).
  3. Lets you play back the ORIGINAL and the CLEANED audio side by side.
  4. Shows objective SNR/quality numbers and downloadable output + spectrograms.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from infer import denoise_file, FINETUNED_PATH  # noqa: E402

st.set_page_config(page_title="AI-ANC-Defense — Military Noise Filter", layout="centered")

st.title("🎙️ AI-ANC-Defense")
st.caption("Upload noisy audio (gunshots, helicopter rotor, artillery, vehicle engine, sirens) "
           "and get the clean speech back.")

model_status = "fine-tuned on military noise" if FINETUNED_PATH.exists() else "general pretrained model (no fine-tune found yet)"
st.info(f"Model in use: **{model_status}**")

uploaded = st.file_uploader("Upload a noisy audio file", type=["wav", "mp3", "flac", "m4a"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / uploaded.name
        in_path.write_bytes(uploaded.getbuffer())

        out_path = Path(tmp) / "enhanced.wav"

        with st.spinner("Filtering out defence noise..."):
            denoise_file(str(in_path), str(out_path),
                         use_finetuned=FINETUNED_PATH.exists())

        clean_wav, sr = sf.read(out_path)
        noisy_wav, noisy_sr = librosa.load(in_path, sr=sr, mono=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔊 Original (noisy)")
            st.audio(str(in_path))
        with col2:
            st.subheader("✅ Cleaned (AI-filtered)")
            out_bytes = out_path.read_bytes()
            st.audio(out_bytes, format="audio/wav")
            st.download_button("Download cleaned audio", data=out_bytes,
                                file_name="cleaned_" + uploaded.name.rsplit(".", 1)[0] + ".wav",
                                mime="audio/wav")

        # --- Objective metrics (best-effort; requires matching length) ---
        st.subheader("📊 Quality metrics")
        try:
            from pystoi import stoi
            n = min(len(noisy_wav), len(clean_wav))
            noisy_trim, clean_trim = noisy_wav[:n], clean_wav[:n]
            snr_before = 10 * np.log10(np.sum(noisy_trim ** 2) / (np.sum((noisy_trim - clean_trim) ** 2) + 1e-8))
            energy_reduction_db = 10 * np.log10(
                (np.mean(noisy_trim ** 2) + 1e-12) / (np.mean(clean_trim ** 2) + 1e-12)
            )
            c1, c2 = st.columns(2)
            c1.metric("Noise energy reduced by", f"{max(energy_reduction_db, 0):.1f} dB")
            c2.metric("Output length", f"{len(clean_wav)/sr:.1f} s")
            st.caption("Note: a true STOI/PESQ score needs a *clean reference* recording, which "
                       "isn't available for an arbitrary uploaded file — these numbers describe "
                       "how much the background energy dropped, not intelligibility.")
        except Exception as e:
            st.warning(f"Could not compute extended metrics: {e}")

        # --- Spectrograms: the clearest visual proof the model worked ---
        st.subheader("🌈 Spectrograms (before vs after)")
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        for ax, wav, title in [(axes[0], noisy_wav, "Original (noisy)"),
                                (axes[1], clean_wav, "Cleaned (AI-filtered)")]:
            D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
            ax.set_title(title)
        fig.tight_layout()
        st.pyplot(fig)

else:
    st.write("Upload a file above to get started. Try a clip of speech recorded near "
             "a gunshot, helicopter, or engine noise.")

st.divider()
st.caption("AI-ANC-Defense — hybrid AI/ML adaptive noise cancellation prototype for defence "
           "communication (SIH26052). Model: fine-tuned Facebook Denoiser (time-domain, "
           "waveform-to-waveform).")
