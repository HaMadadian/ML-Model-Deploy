import os
from pathlib import Path
from collections import defaultdict

import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# -------------------------------------------------
# Configuration
# -------------------------------------------------
RECORDINGS_FOLDER = "my_recordings"
CHUNK_SECONDS = 10
HOP_SECONDS = 5
TOP_K = 10

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Keywords we care about
TARGET_KEYWORDS = [
    "car", "truck", "vehicle", "automobile",
    "aircraft", "airplane", "plane", "helicopter",
    "motorcycle", "bus", "engine", "traffic"
]

# -------------------------------------------------
# Load model
# -------------------------------------------------
print("Loading AST model... (this may take a moment the first time)")
device = "cuda" if torch.cuda.is_available() else "cpu"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Get label names
id2label = model.config.id2label
print(f"Model loaded on: {device}")
print(f"Number of classes: {len(id2label)}\n")


def load_audio(file_path, target_sr=16000):
    """Load audio and resample to 16 kHz (AST default)"""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def split_audio(audio, sr, chunk_sec=10, hop_sec=5):
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


def classify_chunk(audio_chunk, sr):
    """Run AST on one chunk"""
    inputs = feature_extractor(
        audio_chunk,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    return probs


def classify_file(file_path):
    print(f"\nProcessing: {file_path}")
    audio, sr = load_audio(file_path)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f} seconds")

    chunks = split_audio(audio, sr, CHUNK_SECONDS, HOP_SECONDS)
    print(f"  Split into {len(chunks)} chunks")

    # Average probabilities across chunks
    score_sum = np.zeros(len(id2label))
    
    for chunk in chunks:
        probs = classify_chunk(chunk, sr)
        score_sum += probs

    avg_scores = score_sum / len(chunks)

    # Create ranked list
    ranked = []
    for i, score in enumerate(avg_scores):
        label = id2label[i]
        ranked.append((label, float(score)))

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

    print("\n  Top predictions:")
    for label, score in ranked[:TOP_K]:
        is_target = any(keyword in label.lower() for keyword in TARGET_KEYWORDS)
        marker = " ← TARGET" if is_target else ""
        bar = "█" * int(score * 40)
        print(f"  {score:.3f}  {bar:<40}  {label}{marker}")

    return ranked


def main():
    folder = Path(RECORDINGS_FOLDER)

    if not folder.exists():
        print(f"Folder '{RECORDINGS_FOLDER}' does not exist.")
        return

    extensions = {".wav", ".mp3", ".flac", ".ogg"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not files:
        print(f"No supported audio files found in '{RECORDINGS_FOLDER}'.")
        return

    print(f"Found {len(files)} audio file(s)")
    print("=" * 70)

    for file in files:
        try:
            classify_file(str(file))
        except Exception as e:
            print(f"  Error processing {file.name}: {e}")
        print("=" * 70)


if __name__ == "__main__":
    main()