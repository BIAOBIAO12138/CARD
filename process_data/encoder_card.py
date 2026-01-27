import os
import sys
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

def iter_image_paths(image_dir, exts={".jpg",".jpeg",".png",".bmp",".webp",".tiff"}):
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                yield os.path.join(root, f)

def load_image_safe(path):
    img = Image.open(path).convert("RGB")
    return img

def encode_folder_to_parquet(
    image_dir: str,
    out_parquet: str,
    ckpt: str = "google/siglip2-so400m-patch16-512",
    batch_size: int = 32,
    device: str = None,
):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    load_kwargs = dict(local_files_only=True, trust_remote_code=True)
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(ckpt, **load_kwargs).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True, trust_remote_code=True)

    image_paths = list(iter_image_paths(image_dir))
    if len(image_paths) == 0:
        print(f"No images found under: {image_dir}")
        return

    embeddings = []
    ids = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding"):
            batch_paths = image_paths[i:i+batch_size]
            batch_imgs = []
            keep_idx = []
            for j, p in enumerate(batch_paths):
                try:
                    batch_imgs.append(load_image_safe(p))
                    keep_idx.append(j)
                except Exception as e:
                    print(f"[WARN] failed to load {p}: {e}")

            if len(batch_imgs) == 0:
                continue

            inputs = processor(images=batch_imgs, return_tensors="pt")
            model_dtype = next(model.parameters()).dtype
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device=device, dtype=model_dtype)
            feats = model.get_image_features(**inputs)
            feats = feats.detach().cpu().numpy()

            for j, vec in zip(keep_idx, feats):
                embeddings.append(vec.tolist())
                fname = os.path.basename(batch_paths[j])
                stem, _ = os.path.splitext(fname)
                try:
                    item_id = int(stem)
                except ValueError:
                    item_id = stem
                ids.append(item_id)

    df = pd.DataFrame({"ItemID": ids, "embedding": embeddings})
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Saved {len(df)} embeddings to {out_parquet}")
    print(f"Example shape: {len(df.iloc[0]['embedding'])} dims")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", nargs="?", default="images")
    parser.add_argument("out_parquet", nargs="?", default="item_emb_image.parquet")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--ckpt",
        type=str,
        default="google/siglip2-so400m-patch16-512",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0, cuda:5, cpu")
    args = parser.parse_args()

    encode_folder_to_parquet(
        args.image_dir,
        args.out_parquet,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        device=args.device,
    )








    