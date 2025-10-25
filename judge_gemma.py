from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from pathlib import Path
import torch
import json
from tqdm import tqdm
import os

def judge_gemma(model_id="google/gemma-3n-e2b-it"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    folders = input("Enter input folder paths (comma-separated): ").strip().split(",")
    folders = [f.strip() for f in folders]
    output_names = input("Enter output filenames for each folder (comma-separated, no extension): ").strip().split(",")
    output_names = [o.strip() + ".jsonl" for o in output_names]

    if len(folders) != len(output_names):
        raise ValueError("Number of folders and output filenames must match.")

    generated_files = []

    for folder, output_name in zip(folders, output_names):
        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Skipping invalid folder: {folder}")
            continue

        audio_files = [f for f in folder_path.glob("*.mp3")] + \
                      [f for f in folder_path.glob("*.ogg")] + \
                      [f for f in folder_path.glob("*.m4a")] + \
                      [f for f in folder_path.glob("*.wav")]

        if not audio_files:
            print(f"No audio files found in {folder}, skipping.")
            continue

        with open(output_name, "a", encoding="utf-8") as out:
            for audio_file in tqdm(audio_files, desc=f"Processing {folder}"):
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant, who is an expert in identification of Human Vs AI sung songs."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": str(audio_file)},
                            {"type": "text", "text": "Do you think this song is sung by a human or generated using an AI? Answer in Yes/No, followed by 3 sentence explanation."}
                        ]
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device, dtype=torch.float16)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(**inputs, max_new_tokens=300, do_sample=False)
                    generation = generation[0][input_len:]

                decoded = processor.decode(generation, skip_special_tokens=True)

                record = {
                    "filepath": str(audio_file),
                    "response": decoded.strip()
                }
                out.write(json.dumps(record) + "\n")

        generated_files.append(output_name)
        print(f"Saved results for {folder} → {output_name}")

    print("\nUploading generated JSONL files to Hugging Face dataset...")
    for f in generated_files:
        os.system(f"hf upload sleeping-ai/Gemma-Judge {f} --repo-type=dataset")
    print("Upload complete: sleeping-ai/Gemma-Judge")

if __name__ == "__main__":
    judge_gemma()
