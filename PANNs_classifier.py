import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
from panns_inference import AudioTagging, labels

# -------------------------------------------------
# Configuration
# -------------------------------------------------
RECORDINGS_FOLDER = "my_recordings"
CHUNK_SECONDS = 10
HOP_SECONDS = 5
TOP_K = 10          # Show top 10 predictions

# We care most about these classes (AudioSet names)
TARGET_KEYWORDS = [
    "car", "truck", "vehicle", "automobile",
    "aircraft", "airplane", "plane", "helicopter",
    "motorcycle", "bus", "engine", "traffic"
]

# -------------------------------------------------
# Load model
# -------------------------------------------------
print("Loading PANNs (CNN14) model...")
at = AudioTagging(
    checkpoint_path=r"C:\Users\hamedmadadian\panns_data\Cnn14_mAP=0.431.pth",
    device="cuda" if __import__("torch").cuda.is_available() else "cpu"
)
print("Model loaded.\n")


def load_audio(file_path, target_sr=32000):
    """Load and resample audio to 32 kHz (PANNs default)"""
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


def classify_file(file_path):
    print(f"\nProcessing: {file_path}")
    audio, sr = load_audio(file_path)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f} seconds")

    chunks = split_audio(audio, sr, CHUNK_SECONDS, HOP_SECONDS)
    print(f"  Split into {len(chunks)} chunks")

    # Collect scores from all chunks
    score_sum = defaultdict(float)
    count = defaultdict(int)

    for chunk in chunks:
        # panns_inference expects shape (batch, samples)
        clipwise_output, _ = at.inference(chunk[None, :])
        clipwise_output = clipwise_output[0]  # shape: (527,)

        for i, score in enumerate(clipwise_output):
            label = labels[i]
            score_sum[label] += float(score)
            count[label] += 1

    # Average scores
    avg_scores = {label: score_sum[label] / count[label] for label in score_sum}

    # Sort by score
    ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n  Top predictions:")
    for i, (label, score) in enumerate(ranked[:TOP_K]):
        # Highlight relevant classes
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
        print("Supported formats: .wav, .mp3, .flac, .ogg")
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