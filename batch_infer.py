import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ====== Configurable parameters ======
# Path to the pretrained model directory (replace with your model path)
MODEL_PATH = "XXX"  # e.g., "/path/to/Qwen2.5-VL-7B-Instruct"
# Root directory of your dataset (replace with your dataset root)
DATA_ROOT = Path("XXX")  # e.g., "/path/to/dataset"
# Directory containing encoder text files
ENC_DIR = DATA_ROOT / "encoder_text"
# Directory containing input images
IMG_DIR = DATA_ROOT / "image_revised"
# Directory to write output descriptions
OUTPUT_PATH = Path("XXX")  # e.g., "/path/to/output"
# Path to error log file
ERROR_LOG = Path("XXX")  # e.g., "/path/to/error_log.txt"

# ====== Command-line argument parsing ======
parser = argparse.ArgumentParser(description="Batch inference for image-text pairs")
parser.add_argument("--start-id", type=int, default=None,
                    help="Starting file ID to process (integer, no leading zeros)")
parser.add_argument("--end-id",   type=int, default=None,
                    help="Ending file ID to process (integer, no leading zeros)")
args = parser.parse_args()

# ====== Load model and processor ======
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model.eval()

# ====== Prompt template builder ======
def build_prompt(feature_log: str) -> str:
    return (
        "The uploaded image and text are from the same CAD model. "
        "Please incorporate both parameter and feature details from the input text and observations from multiple view images when describing its appearance, shape, and function in English. "
        "Divide the description into two paragraphs: Appearance first, then Function.\n"
        "For example:\n"
        "[Appearance]: ...\n"
        "[Function]: ...\n\n"
        "Include key visual details like dimensions, shapes, and characteristics from the text.\n\n"
        "Now describe the following model features:\n\n"
        f"{feature_log}"
    )

# ====== Single-sample processing ======
def process_sample(file_id: str):
    out_file = OUTPUT_PATH / f"{file_id}.txt"
    if out_file.exists():
        return  # Skip if already processed

    try:
        txt_path = ENC_DIR / f"{file_id}.txt"
        img_path = IMG_DIR / f"{file_id}.png"

        # Verify input files exist
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing text file: {txt_path}")
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image file: {img_path}")

        # Load text and image
        feature_log = txt_path.read_text(encoding="utf-8", errors="ignore")
        image = Image.open(img_path).convert("RGB")

        # Prepare chat-style inputs
        prompt = build_prompt(feature_log)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "path": str(img_path)},
                {"type": "text",  "text": prompt}
            ]}
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.8,
                top_p=0.9,
            )

        # Decode and save result
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        result = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        out_file.write_text(result, encoding="utf-8")

    except Exception as e:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"[File {file_id}] Error: {e}\n")

# ====== Main entrypoint ======
if __name__ == "__main__":
    # Gather all text sample IDs
    all_txt = sorted(ENC_DIR.glob("*.txt"))
    file_ids = [p.stem for p in all_txt]

    # Apply --start-id/--end-id filters
    if args.start_id is not None:
        file_ids = [fid for fid in file_ids if int(fid) >= args.start_id]
    if args.end_id is not None:
        file_ids = [fid for fid in file_ids if int(fid) <= args.end_id]

    print(f"🚀 Found {len(file_ids)} samples, starting inference...")
    for fid in tqdm(file_ids, desc="Samples", ncols=80):
        process_sample(fid)
    print("✅ All inference completed!")
