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
output_names = [o.strip() for o in output_names]

if len(input_folders) != len(output_names):
    raise ValueError("Number of input folders and output filenames must match.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_audio(file, sr=None):
    y, sr = librosa.load(file, sr=sr, mono=True)
    waveform = torch.tensor(y).unsqueeze(0).to(device)
    return waveform, sr

def get_middle_segment(y, sr, target_duration):
    target_len = int(target_duration * sr)
    if len(y) <= target_len:
        return y
    start = (len(y) - target_len) // 2
    end = start + target_len
    return y[start:end]

for folder, base_output_name in zip(input_folders, output_names):
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

        for mode in ["full", "middle"]:
            fake_scores = []
            output_name = f"{base_output_name}_{mode}.jsonl"

            with open(output_name, "a", encoding="utf-8") as f:
                for file in tqdm(audio_files, desc=f"{mode.upper()} | {model_name}"):
                    try:
                        y, sr = librosa.load(file, sr=None, mono=True)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load {file}: {e}")

                    if mode == "middle":
                        if "120s" in model_name:
                            y = get_middle_segment(y, sr, 120)
                        elif "5s" in model_name:
                            y = get_middle_segment(y, sr, 5)

                    waveform = torch.tensor(y).unsqueeze(0).to(device)

                    with torch.no_grad():
                        predictions = model(waveform)
                        probs = torch.sigmoid(predictions).squeeze()
                        prob_val = probs.item()
                        pred_class = int(torch.round(probs).item())
                        label = "FAKE" if pred_class == 1 else "HUMAN"

                    fake_scores.append(prob_val)

                    record = {
                        "model": model_name,
                        "filepath": str(file),
                        "pred_class": label,
                        "model_prob": prob_val
                    }
                    f.write(json.dumps(record) + "\n")

            avg_fake = np.mean(fake_scores) * 100
            avg_human = 100 - avg_fake
            print(f"\n[LOG] Model {model_name} ({mode}) finished for folder '{folder}':")
            print(f"   → Avg Fake: {avg_fake:.2f}%")
            print(f"   → Avg Human: {avg_human:.2f}%\n")

            print(f"Uploading {output_name} to Hugging Face dataset...")
            os.system(f"hf upload sleeping-ai/SONICS-Humanfakebench {output_name} --repo-type=dataset")
            print(f"Finished uploading {output_name}\n")
