import os
import shutil

def extract_human(input_folder="iclr-human", output_folder="human-music"):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.endswith(".m4a"):
                src = os.path.join(root, f)
                dst = os.path.join(output_folder, f.lstrip("-"))
                shutil.copy2(src, dst)
    print("All .m4a files copied to", output_folder)
