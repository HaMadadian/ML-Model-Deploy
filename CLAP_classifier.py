import os
from pathlib import Path

import torch
import librosa
import numpy as np
from transformers import ClapModel, ClapProcessor

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_NAME = "laion/clap-htsat-fused"
CHUNK_SECONDS = 10          # Process audio in 10-second chunks
HOP_SECONDS = 5             # Overlap between chunks
RECORDINGS_FOLDER = "my_recordings"   # <-- automatic folder

# Labels we care about
LABELS = [
    "a car passing by",
    "a truck passing by",
    "an airplane or aircraft flying",
    "an airplane or aircraft flying overhead",
    "a train passing by",
    "a motorcycle",
    "background noise",
    "human speech",
    "wind noise",
    "rain",
    "bird"
]

# -------------------------------------------------
# Load model
# -------------------------------------------------
print("Loading CLAP model... (this may take a moment the first time)")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = ClapProcessor.from_pretrained(MODEL_NAME)
model = ClapModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"Model loaded on: {device}\n")


def load_audio(file_path: str, target_sr: int = 48000):
    """Load audio and resample to 48 kHz"""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def split_audio(audio: np.ndarray, sr: int, chunk_sec: float, hop_sec: float):
    """Split long audio into overlapping chunks"""
    chunk_samples = int(chunk_sec * sr)
    hop_samples = int(hop_sec * sr)

    chunks = []
    for start in range(0, len(audio), hop_samples):
        end = start + chunk_samples
        chunk = audio[start:end]

        if len(chunk) < chunk_samples // 2:
            break

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        chunks.append(chunk)
    return chunks


def classify_chunk(audio_chunk: np.ndarray, sr: int, labels: list):
    """Run CLAP on one audio chunk"""
    inputs = processor(
        text=labels,
        audio=audio_chunk,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_audio
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return dict(zip(labels, probs))


def classify_file(file_path: str):
    """Full pipeline for one audio file"""
    print(f"\nProcessing: {file_path}")
    audio, sr = load_audio(file_path)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f} seconds")

    chunks = split_audio(audio, sr, CHUNK_SECONDS, HOP_SECONDS)
    print(f"  Split into {len(chunks)} chunks")

    all_scores = {label: [] for label in LABELS}

    for chunk in chunks:
        scores = classify_chunk(chunk, sr, LABELS)
        for label, score in scores.items():
            all_scores[label].append(score)

    # Average score across all chunks
    final_scores = {
        label: float(np.mean(scores)) for label, scores in all_scores.items()
    }

    # Sort by confidence (highest first)
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n  Results (highest confidence first):")
    for label, score in ranked:
        bar = "█" * int(score * 30)
        print(f"  {score:.3f}  {bar:<30}  {label}")

    return ranked


def main():
    folder = Path(RECORDINGS_FOLDER)

    if not folder.exists():
        print(f"Folder '{RECORDINGS_FOLDER}' does not exist.")
        print("Please create it and put your audio files inside.")
        return

    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not files:
        print(f"No audio files found in '{RECORDINGS_FOLDER}'.")
        return

    print(f"Found {len(files)} audio file(s) in '{RECORDINGS_FOLDER}'")
    print("=" * 65)

    for file in files:
        classify_file(str(file))
        print("=" * 65)


if __name__ == "__main__":
    main()