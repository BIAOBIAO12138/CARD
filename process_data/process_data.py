import argparse
import os
import gzip
import json
import ast
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from tqdm import tqdm


def iter_json_like(path):
    def _parse_line(s):
        s = s.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return None

    if path.endswith(".gz"):
        with gzip.open(path, "rb") as g:
            for l in g:
                try:
                    yield ast.literal_eval(l.decode("utf-8"))
                except Exception:
                    try:
                        yield json.loads(l)
                    except Exception:
                        continue
    else:
        with open(path, "r") as f:
            head = f.read(1)
            f.seek(0)
            if head == "[":
                txt = f.read()
                try:
                    arr = json.loads(txt)
                except Exception:
                    arr = ast.literal_eval(txt)
                for obj in arr:
                    yield obj
            else:
                for line in f:
                    obj = _parse_line(line)
                    if obj is not None:
                        yield obj


def write_json_from_any(in_path, out_json_path):
    with open(out_json_path, "w") as f:
        for obj in iter_json_like(in_path):
            f.write(json.dumps(obj) + "\n")


def build_user_seq(reviews_iter) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, List[Tuple[int, int]]]]:
    user_map: Dict[str, int] = {}
    item_map: Dict[str, int] = {}
    user_seq: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for r in reviews_iter:
        if not isinstance(r, dict):
            continue
        uid_raw = r.get("reviewerID")
        iid_raw = r.get("asin")
        ts = r.get("unixReviewTime")
        if uid_raw is None or iid_raw is None or ts is None:
            continue
        if uid_raw not in user_map:
            user_map[uid_raw] = len(user_map) + 1
        if iid_raw not in item_map:
            item_map[iid_raw] = len(item_map) + 1
        uid = user_map[uid_raw]
        iid = item_map[iid_raw]
        try:
            ts_int = int(ts)
        except Exception:
            ts_int = 0
        user_seq[uid].append((iid, ts_int))

    return user_map, item_map, user_seq


def _build_http_session(timeout: int = 10, retries: int = 3, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    })
    s.request_timeout = timeout
    return s


def _iter_meta_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                try:
                    yield ast.literal_eval(line)
                except Exception:
                    continue


def _first_url(rec):
    url = rec.get("imageURLHighRes") or rec.get("imageURL") or rec.get("imUrl")
    if isinstance(url, list):
        return url[0] if url else None
    return url


def download_images_for_interacted_items(
    meta_json_path: str,
    item_map: Dict[str, int],
    images_dir: str,
    timeout: int,
    retries: int,
    backoff: float,
    workers: int,
) -> Set[int]:
    os.makedirs(images_dir, exist_ok=True)

    asin2url: Dict[str, str] = {}
    for rec in _iter_meta_jsonl(meta_json_path):
        a = rec.get("asin")
        if a in item_map:
            u = _first_url(rec)
            if u and a not in asin2url:
                asin2url[a] = u

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _download_one(job):
        asin, url, save_path = job
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        for attempt in range(max(1, retries + 1)):
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                r.raise_for_status()
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img.save(save_path)
                return True
            except Exception:
                if attempt == retries:
                    return False
        return False

    jobs = []
    for asin, url in asin2url.items():
        item_id = item_map[asin]
        save_path = os.path.join(images_dir, f"{int(item_id)}.jpg")
        if os.path.exists(save_path):
            continue
        jobs.append((asin, url, save_path))

    failed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_download_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            ok = fut.result()
            if not ok:
                failed.append(1)

    item_ids_with_image: Set[int] = set()
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        name, _ = os.path.splitext(fname)
        try:
            iid = int(name)
        except Exception:
            continue
        item_ids_with_image.add(iid)

    return item_ids_with_image


def apply_5core(user_seq: Dict[int, List[Tuple[int, int]]], min_interactions: int = 5) -> Tuple[Dict[int, List[Tuple[int, int]]], Set[int]]:
    user_deg: Dict[int, int] = {}
    item_deg: Dict[int, int] = defaultdict(int)
    item2users: Dict[int, Set[int]] = defaultdict(set)

    for uid, seq in user_seq.items():
        user_deg[uid] = len(seq)
        for iid, _ in seq:
            item_deg[iid] += 1
            item2users[iid].add(uid)

    removed_users: Set[int] = set()
    removed_items: Set[int] = set()

    changed = True
    while changed:
        changed = False
        for uid, deg in list(user_deg.items()):
            if uid in removed_users:
                continue
            if deg < min_interactions:
                removed_users.add(uid)
                changed = True
                for iid, _ in user_seq.get(uid, []):
                    if iid in removed_items:
                        continue
                    if item_deg.get(iid, 0) > 0:
                        item_deg[iid] -= 1

        for iid, deg in list(item_deg.items()):
            if iid in removed_items:
                continue
            if deg < min_interactions:
                removed_items.add(iid)
                changed = True
                for uid in item2users.get(iid, set()):
                    if uid in removed_users:
                        continue
                    if user_deg.get(uid, 0) > 0:
                        user_deg[uid] -= 1

    kept_items: Set[int] = set()
    user_seq_filtered: Dict[int, List[Tuple[int, int]]] = {}

    for uid, seq in user_seq.items():
        if uid in removed_users:
            continue
        new_seq = [(iid, ts) for (iid, ts) in seq if iid not in removed_items]
        if len(new_seq) < min_interactions:
            continue
        new_seq.sort(key=lambda x: x[1])
        user_seq_filtered[uid] = new_seq
        for iid, _ in new_seq:
            kept_items.add(iid)

    return user_seq_filtered, kept_items


def build_splits_from_user_seq(user_seq: Dict[int, List[Tuple[int, int]]]):
    train_rows = []
    valid_rows = []
    test_rows = []

    for uid, seq in user_seq.items():
        item_seq = [iid for (iid, _) in seq]
        n = len(item_seq)
        if n < 2:
            continue
        if n >= 3:
            train_seq = item_seq[:-2]
            if len(train_seq) >= 2:
                train_rows.append({"history": train_seq[:-1], "target": train_seq[-1]})
        val_seq = item_seq[:-1]
        if len(val_seq) >= 2:
            valid_rows.append({"history": val_seq[:-1], "target": val_seq[-1]})
        if n >= 2:
            test_rows.append({"history": item_seq[:-1], "target": item_seq[-1]})

    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, valid_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Beauty")
    parser.add_argument("--reviews_gz", type=str, required=True)
    parser.add_argument("--meta_gz", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--sent_model",
        type=str,
        default="sentence-transformers/sentence-t5-base",
    )
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument("--images_out_dir", type=str, default=None)
    parser.add_argument("--download_timeout", type=int, default=10)
    parser.add_argument("--download_retries", type=int, default=3)
    parser.add_argument("--download_backoff", type=float, default=0.5)
    parser.add_argument("--download_workers", type=int, default=16)
    parser.add_argument("--skip_text_embedding", action="store_true",
                        help="If set, skip the final text embedding encoding step.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    reviews_json_path = os.path.join(args.out_dir, f"{args.dataset_name}.json")
    write_json_from_any(args.reviews_gz, reviews_json_path)

    user_map, item_map, user_seq = build_user_seq(iter_json_like(args.reviews_gz))

    np.save(os.path.join(args.out_dir, "user_mapping.npy"), user_map)
    np.save(os.path.join(args.out_dir, "item_mapping.npy"), item_map)

    meta_json_path = os.path.join(args.out_dir, f"{args.dataset_name}_metadata.json")
    write_json_from_any(args.meta_gz, meta_json_path)

    images_dir = args.images_out_dir or os.path.join(args.out_dir, "images")
    print("[INFO] Downloading images for interacted items...")
    item_ids_with_image = download_images_for_interacted_items(
        meta_json_path=meta_json_path,
        item_map=item_map,
        images_dir=images_dir,
        timeout=args.download_timeout,
        retries=args.download_retries,
        backoff=args.download_backoff,
        workers=args.download_workers,
    )
    print(f"[INFO] Items with downloaded images: {len(item_ids_with_image)}")

    user_seq_with_image: Dict[int, List[Tuple[int, int]]] = {}
    for uid, seq in user_seq.items():
        filtered = [(iid, ts) for (iid, ts) in seq if iid in item_ids_with_image]
        if not filtered:
            continue
        filtered.sort(key=lambda x: x[1])
        seen_items: Set[int] = set()
        dedup_seq: List[Tuple[int, int]] = []
        for iid, ts in filtered:
            if iid in seen_items:
                continue
            seen_items.add(iid)
            dedup_seq.append((iid, ts))
        if len(dedup_seq) >= 2:
            user_seq_with_image[uid] = dedup_seq

    print(f"[INFO] Users with at least 2 interactions on imaged items (deduped): {len(user_seq_with_image)}")

    print(f"[INFO] Applying {args.min_interactions}-core filtering...")
    user_seq_filtered, kept_items = apply_5core(user_seq_with_image, min_interactions=args.min_interactions)
    print(f"[INFO] Users after {args.min_interactions}-core: {len(user_seq_filtered)}")
    print(f"[INFO] Items after {args.min_interactions}-core: {len(kept_items)}")

    kept_iids_set: Set[int] = set(kept_items)
    removed_images = 0
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        name, _ = os.path.splitext(fname)
        try:
            iid = int(name)
        except Exception:
            continue
        if iid not in kept_iids_set:
            try:
                os.remove(os.path.join(images_dir, fname))
                removed_images += 1
            except Exception:
                continue
    print(f"[INFO] Removed {removed_images} images not in filtered item set")

    print("[INFO] Building splits from filtered interactions...")
    train_df, valid_df, test_df = build_splits_from_user_seq(user_seq_filtered)

    train_out = os.path.join(args.out_dir, "train.parquet")
    valid_out = os.path.join(args.out_dir, "valid.parquet")
    test_out = os.path.join(args.out_dir, "test.parquet")

    train_df.to_parquet(train_out, index=False)
    valid_df.to_parquet(valid_out, index=False)
    test_df.to_parquet(test_out, index=False)

    print(f"[INFO] Saved train to: {train_out} (rows={len(train_df)})")
    print(f"[INFO] Saved valid to: {valid_out} (rows={len(valid_df)})")
    print(f"[INFO] Saved test to:  {test_out} (rows={len(test_df)})")

    if not args.skip_text_embedding:
        print("[INFO] Encoding text semantics for kept items only...")
        itemid_by_asin = {asin: iid for asin, iid in item_map.items()}
        kept_iids_set: Set[int] = set(kept_items)

        item_info = {}
        with open(meta_json_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except Exception:
                    try:
                        m = ast.literal_eval(line)
                    except Exception:
                        continue
                asin = m.get("asin")
                if not asin or asin not in itemid_by_asin:
                    continue
                iid = itemid_by_asin[asin]
                if iid not in kept_iids_set:
                    continue
                item_info[iid] = {
                    "title": m.get("title"),
                    "categories": m.get("categories"),
                    "description": m.get("description"),
                }

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.sent_model)
        iids = list(item_info.keys())
        texts = [
            f"'title':{item_info[iid].get('title', '')}\n 'categories':{item_info[iid].get('categories', '')}\n 'description':{item_info[iid].get('description', '')}"
            for iid in iids
        ]
        embs = model.encode(texts, batch_size=64, show_progress_bar=True)
        rows = [{"ItemID": iid, "embedding": emb.tolist()} for iid, emb in zip(iids, embs)]

        if rows:
            emb_df = pd.DataFrame(rows)
            emb_out = os.path.join(args.out_dir, "item_emb.parquet")
            emb_df.to_parquet(emb_out, index=False)
            print(f"[INFO] Saved text embeddings to: {emb_out} (rows={len(emb_df)})")
    else:
        print("[INFO] Skipping text embedding encoding step as requested.")


if __name__ == "__main__":
    main()
