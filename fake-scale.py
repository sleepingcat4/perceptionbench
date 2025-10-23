import librosa
import torch
from sonics import HFAudioClassifier
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import os

models = [
    "awsaf49/sonics-spectttra-alpha-120s",
    "awsaf49/sonics-spectttra-gamma-120s",
    "awsaf49/sonics-spectttra-beta-120s",
    "awsaf49/sonics-spectttra-gamma-5s",
    "awsaf49/sonics-spectttra-beta-5s",
    "awsaf49/sonics-spectttra-alpha-5s"
]

input_folders = input("Enter input folder paths (comma-separated): ").strip().split(",")
input_folders = [f.strip() for f in input_folders]

output_names = input("Enter output filenames for each folder (comma-separated, no extension needed): ").strip().split(",")
output_names = [o.strip() + ".jsonl" if not o.strip().endswith(".jsonl") else o.strip() for o in output_names]

if len(input_folders) != len(output_names):
    raise ValueError("Number of input folders and output filenames must match.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_audio(file, sr=None):
    y, sr = librosa.load(file, sr=sr, mono=True)
    waveform = torch.tensor(y).unsqueeze(0).to(device)
    return waveform, sr

for folder, output_name in zip(input_folders, output_names):
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Skipping invalid folder: {folder}")
        continue

    audio_files = list(folder_path.glob("*.[om][gp][3g]"))
    if not audio_files:
        print(f"No audio files found in {folder}, skipping.")
        continue

    for model_name in models:
        print(f"\n=== USING MODEL: {model_name} ===\n")
        model = HFAudioClassifier.from_pretrained(model_name).to(device)
        model.eval()

        fake_scores = []
        with open(output_name, "a", encoding="utf-8") as f:
            for file in tqdm(audio_files, desc=f"Processing with {model_name}"):
                try:
                    waveform, sr = load_audio(file)
                except Exception as e:
                    raise RuntimeError(f"Failed to load {file}: {e}")

                with torch.no_grad():
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
        print(f"\n[LOG] Model {model_name} finished for folder '{folder}':")
        print(f"   → Avg Fake: {avg_fake:.2f}%")
        print(f"   → Avg Human: {avg_human:.2f}%\n")

    # Upload to Hugging Face after processing this folder
    print(f"Uploading {output_name} to Hugging Face dataset...")
    os.system(f"hf upload sleeping-ai/SONICS-Humanfakebench {output_name} --repo-type=dataset")
    print(f"Finished uploading {output_name}\n")
