import os
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import librosa
from transformers import ClapModel, ClapProcessor, AutoFeatureExtractor, AutoModelForAudioClassification

# =========================================================
# ================== CONFIGURABLE SECTION =================
# =========================================================

RECORDINGS_FOLDER = "my_recordings"
OUTPUT_EXCEL = "classification_results.xlsx"

CHUNK_SECONDS = 10.0
HOP_SECONDS = 5.0

# Decision thresholds (you can tweak these later)
HIGH_SECONDS = 5.0
MEDIUM_SECONDS = 3.0

HIGH_SCORE = 0.45
MEDIUM_SCORE = 0.30
LOW_SCORE = 0.20

# Common target classes
TARGET_CLASSES = ["car", "truck", "aircraft", "train", "vehicle"]

# =========================================================
# ====================== MODEL LOADING ====================
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ----- CLAP -----
print("Loading CLAP...")
CLAP_MODEL_NAME = "laion/clap-htsat-fused"
clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
clap_model = ClapModel.from_pretrained(CLAP_MODEL_NAME).to(device)
clap_model.eval()

CLAP_LABELS = [
    "a car passing by",
    "a truck passing by",
    "an airplane or aircraft flying",
    "a train passing by",
    "a vehicle"
]

# ----- AST -----
print("Loading AST...")
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
ast_processor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
ast_model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_NAME).to(device)
ast_model.eval()
ast_id2label = ast_model.config.id2label

# ----- PANNs -----
print("Loading PANNs...")
try:
    from panns_inference import AudioTagging, labels as panns_labels
    panns_at = AudioTagging(checkpoint_path=None, device=device)
    PANNS_AVAILABLE = True
except Exception as e:
    print(f"PANNs could not be loaded: {e}")
    print("Continuing with CLAP + AST only.")
    PANNS_AVAILABLE = False

print("\nAll available models loaded.\n")

# =========================================================
# ====================== HELPERS ==========================
# =========================================================

def load_audio(path, sr):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def split_audio(audio, sr, chunk_sec, hop_sec):
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


def map_to_common(label: str) -> str | None:
    label = label.lower()
    if "truck" in label or "lorry" in label:
        return "truck"
    if "car" in label or "automobile" in label:
        return "car"
    if "train" in label:
        return "train"
    if any(x in label for x in ["aircraft", "airplane", "plane", "jet", "helicopter", "propeller"]):
        return "aircraft"
    if "vehicle" in label or "engine" in label or "bus" in label or "motorcycle" in label:
        return "vehicle"
    return None


def decide(max_score, presence_seconds):
    if max_score >= HIGH_SCORE and presence_seconds >= HIGH_SECONDS:
        return "High"
    if max_score >= MEDIUM_SCORE and presence_seconds >= MEDIUM_SECONDS:
        return "Medium"
    if max_score >= LOW_SCORE:
        return "Low"
    return "Not detected"


# =========================================================
# ====================== MODEL INFERENCE ==================
# =========================================================

def run_clap(chunks, sr):
    results = {cls: [] for cls in TARGET_CLASSES}

    for chunk in chunks:
        inputs = clap_processor(
            text=CLAP_LABELS,
            audio=chunk,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clap_model(**inputs)
            probs = torch.softmax(outputs.logits_per_audio, dim=-1)[0].cpu().numpy()

        mapping = {
            "a car passing by": "car",
            "a truck passing by": "truck",
            "an airplane or aircraft flying": "aircraft",
            "a train passing by": "train",
            "a vehicle": "vehicle"
        }
        for label, prob in zip(CLAP_LABELS, probs):
            common = mapping[label]
            results[common].append(float(prob))

    return results


def run_ast(chunks, sr):
    results = {cls: [] for cls in TARGET_CLASSES}

    for chunk in chunks:
        inputs = ast_processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = ast_model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        scores = defaultdict(list)
        for i, p in enumerate(probs):
            common = map_to_common(ast_id2label[i])
            if common:
                scores[common].append(float(p))

        for cls in TARGET_CLASSES:
            results[cls].append(max(scores[cls]) if scores[cls] else 0.0)

    return results


def run_panns(chunks, sr):
    results = {cls: [] for cls in TARGET_CLASSES}

    for chunk in chunks:
        clipwise, _ = panns_at.inference(chunk[None, :])
        probs = clipwise[0]

        scores = defaultdict(list)
        for i, p in enumerate(probs):
            common = map_to_common(panns_labels[i])
            if common:
                scores[common].append(float(p))

        for cls in TARGET_CLASSES:
            results[cls].append(max(scores[cls]) if scores[cls] else 0.0)

    return results


# =========================================================
# ====================== MAIN PIPELINE ====================
# =========================================================

def process_file(file_path):
    print(f"\nProcessing: {file_path.name}")

    # Load at different sample rates
    audio_clap = load_audio(file_path, 48000)
    audio_ast = load_audio(file_path, 16000)
    audio_panns = load_audio(file_path, 32000) if PANNS_AVAILABLE else None

    chunks_clap = split_audio(audio_clap, 48000, CHUNK_SECONDS, HOP_SECONDS)
    chunks_ast = split_audio(audio_ast, 16000, CHUNK_SECONDS, HOP_SECONDS)
    chunks_panns = split_audio(audio_panns, 32000, CHUNK_SECONDS, HOP_SECONDS) if PANNS_AVAILABLE else []

    print(f"  Chunks: CLAP={len(chunks_clap)}, AST={len(chunks_ast)}, PANNs={len(chunks_panns)}")

    # Run models
    clap_raw = run_clap(chunks_clap, 48000)
    ast_raw = run_ast(chunks_ast, 16000)
    panns_raw = run_panns(chunks_panns, 32000) if PANNS_AVAILABLE else {cls: [0.0] for cls in TARGET_CLASSES}

    # Aggregate
    def aggregate(raw_scores):
        output = {}
        for cls in TARGET_CLASSES:
            scores = raw_scores[cls]
            if not scores:
                output[cls] = {"max_score": 0.0, "presence_seconds": 0.0, "decision": "Not detected"}
                continue

            max_score = max(scores)
            # Count how many chunks exceeded MEDIUM_SCORE
            active_chunks = sum(1 for s in scores if s >= MEDIUM_SCORE)
            presence_seconds = active_chunks * HOP_SECONDS   # approximate

            decision = decide(max_score, presence_seconds)
            output[cls] = {
                "max_score": round(max_score, 3),
                "presence_seconds": round(presence_seconds, 1),
                "decision": decision
            }
        return output

    clap_res = aggregate(clap_raw)
    ast_res = aggregate(ast_raw)
    panns_res = aggregate(panns_raw)

    # Combined decision
    combined = {}
    for cls in TARGET_CLASSES:
        decisions = [
            clap_res[cls]["decision"],
            ast_res[cls]["decision"],
            panns_res[cls]["decision"]
        ]
        high_or_med = sum(1 for d in decisions if d in ["High", "Medium"])

        if high_or_med >= 2:
            final = "High"
        elif high_or_med == 1:
            final = "Medium"
        elif any(d == "Low" for d in decisions):
            final = "Low"
        else:
            final = "Not detected"

        combined[cls] = final

    return {
        "CLAP": clap_res,
        "AST": ast_res,
        "PANNs": panns_res,
        "Combined": combined
    }


def main():
    folder = Path(RECORDINGS_FOLDER)
    if not folder.exists():
        print(f"Folder '{RECORDINGS_FOLDER}' not found.")
        return

    files = [f for f in folder.iterdir() if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
    if not files:
        print("No audio files found.")
        return

    print(f"Found {len(files)} files\n")

    all_results = {}
    summary_rows = []

    for file in files:
        try:
            result = process_file(file)
            all_results[file.name] = result

            row = {"file": file.name}
            for cls in TARGET_CLASSES:
                row[cls] = result["Combined"][cls]
            summary_rows.append(row)

        except Exception as e:
            print(f"  Error: {e}")

    # ========== Write Excel ==========
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        # Summary sheet
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Detailed sheets
        for filename, result in all_results.items():
            rows = []
            for model_name in ["CLAP", "AST", "PANNs"]:
                for cls in TARGET_CLASSES:
                    info = result[model_name][cls]
                    rows.append({
                        "Model": model_name,
                        "Class": cls,
                        "Max Score": info["max_score"],
                        "Presence Seconds": info["presence_seconds"],
                        "Decision": info["decision"]
                    })

            # Combined row
            for cls in TARGET_CLASSES:
                rows.append({
                    "Model": "COMBINED",
                    "Class": cls,
                    "Max Score": "-",
                    "Presence Seconds": "-",
                    "Decision": result["Combined"][cls]
                })

            sheet_name = filename[:30]  # Excel sheet name limit
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nResults saved to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()