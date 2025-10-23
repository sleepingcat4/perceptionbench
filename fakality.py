import librosa
import torch
from sonics import HFAudioClassifier
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

models = [
    "awsaf49/sonics-spectttra-alpha-120s",
    "awsaf49/sonics-spectttra-gamma-120s",
    "awsaf49/sonics-spectttra-beta-120s",
    "awsaf49/sonics-spectttra-gamma-5s",
    "awsaf49/sonics-spectttra-beta-5s",
    "awsaf49/sonics-spectttra-alpha-5s"
]

input_folder = input("Enter input folder path: ").strip()
output_name = input("Enter output filename (no extension needed): ").strip()
if not output_name.endswith(".jsonl"):
    output_name += ".jsonl"

audio_files = list(Path(input_folder).glob("*.[om][gp][3g]"))
if not audio_files:
    raise FileNotFoundError(f"No audio files found in {input_folder}")

def load_audio(file, sr=None):
    y, sr = librosa.load(file, sr=sr, mono=True)
    waveform = torch.tensor(y).unsqueeze(0)
    return waveform, sr

for model_name in models:
    print(f"\n=== USING MODEL: {model_name} ===\n")
    model = HFAudioClassifier.from_pretrained(model_name)
    model.eval()

    fake_scores = []
    with open(output_name, "a", encoding="utf-8") as f:
        for file in tqdm(audio_files, desc=f"Processing with {model_name}"):
            try:
                waveform, sr = load_audio(file)
            except Exception as e:
                raise RuntimeError(f"Failed to load {file}: {e}")

            predictions = model(waveform)
            probs = torch.sigmoid(predictions).squeeze()
            pred_class = int(torch.round(probs).item())
            label = "FAKE" if pred_class == 1 else "HUMAN"

            fake_scores.append(probs.item())

            record = {
                "model": model_name,
                "filepath": str(file),
                "pred_class": label
            }
            f.write(json.dumps(record) + "\n")

    avg_fake = np.mean(fake_scores) * 100
    avg_human = 100 - avg_fake
    print(f"\n[LOG] Model {model_name} finished:")
    print(f"   → Avg Fake: {avg_fake:.2f}%")
    print(f"   → Avg Human: {avg_human:.2f}%\n")
