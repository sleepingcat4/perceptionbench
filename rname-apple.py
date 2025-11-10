import os
import json

folder_path = "/root/apple_previews"
files = sorted(f for f in os.listdir(folder_path) if f.endswith(".m4a"))

mapping = []

for idx, filename in enumerate(files, 1):
    new_name = f"{idx:05}.m4a"
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
    mapping.append({"original-filename": filename, "modified-filename": new_name})

with open(os.path.join(folder_path, "mapping.jsonl"), "w") as f:
    for item in mapping:
        f.write(json.dumps(item) + "\n")
