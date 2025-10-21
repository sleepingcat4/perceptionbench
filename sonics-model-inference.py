##############
# install the pip package using this: git+https://github.com/awsaf49/sonics.git
# target class 1 means full fake and 0 means human
##############
from sonics import HFAudioClassifier
import torchaudio
import torch
from tqdm import tqdm
import csv
import os

model = HFAudioClassifier.from_pretrained("awsaf49/sonics-spectttra-gamma-5s")
audio_folder = input("Enter folder containing audio files: ")
output_file = input("Enter output CSV filename (e.g., results.csv): ")

audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith((".mp3", ".ogg"))]

results = []

for file in tqdm(audio_files, desc="predicting fakality"):
    waveform, sr = torchaudio.load(file)
    predictions = model(waveform)  
    probs = torch.sigmoid(predictions).squeeze()
    pred_class = int(torch.round(probs).item())
    human_percent = (1 - probs.item()) * 100
    fake_percent = probs.item() * 100
    results.append({
        "filepath": file,
        "pred_class": pred_class,
        "human_percent": human_percent,
        "fake_percent": fake_percent
    })

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filepath", "pred_class", "human_percent", "fake_percent"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"Saved results to {output_file}")
