import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def _read_mapping_csv(path: str) -> list:
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "index" not in reader.fieldnames or "token" not in reader.fieldnames:
            raise ValueError(f"Unexpected mapping csv header: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        return tokens

    max_idx = max(int(r["index"]) for r in rows)
    tokens = [""] * (max_idx + 1)
    for r in rows:
        tokens[int(r["index"])] = r["token"]
    return tokens


FONT_CANDIDIDATES = []


def _load_font(size: int):
    return ImageFont.load_default()


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _wrap_text_by_width(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw):
    if not text:
        return []
    lines = []
    buf = ""
    for ch in str(text):
        test = buf + ch
        if draw.textlength(test, font=font) <= max_width:
            buf = test
        else:
            if buf:
                lines.append(buf)
            buf = ch
    if buf:
        lines.append(buf)
    return lines


def _topk_cosine_neighbors(emb: np.ndarray, topk: int, block: int = 1024) -> np.ndarray:
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape={emb.shape}")
    n = emb.shape[0]
    emb = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    neigh = np.empty((n, topk), dtype=np.int64)

    for s in tqdm(range(0, n, block), desc="TopK cosine", leave=False):
        e = min(s + block, n)
        sims = emb[s:e] @ emb.T
        for i in range(s, e):
            sims[i - s, i] = -np.inf
        idx = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
        row = np.arange(e - s)[:, None]
        idx_sorted = idx[np.argsort(-sims[row, idx], axis=1)]
        neigh[s:e] = idx_sorted

    return neigh


def _filtered_topk_cosine_neighbors(
    emb: np.ndarray,
    eligible: np.ndarray,
    topk: int,
    candidate_k: int,
    block: int = 1024,
) -> np.ndarray:
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape={emb.shape}")
    n = emb.shape[0]
    if eligible.shape[0] != n:
        raise ValueError("eligible mask size mismatch")
    emb = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    candidate_k = int(max(candidate_k, topk))
    candidate_k = int(min(candidate_k, max(1, n - 1)))

    neigh = np.full((n, topk), -1, dtype=np.int64)
    for s in tqdm(range(0, n, block), desc="TopK cosine", leave=False):
        e = min(s + block, n)
        sims = emb[s:e] @ emb.T
        for i in range(s, e):
            sims[i - s, i] = -np.inf
        idx = np.argpartition(-sims, kth=candidate_k - 1, axis=1)[:, :candidate_k]
        row = np.arange(e - s)[:, None]
        order = np.argsort(-sims[row, idx], axis=1)
        idx_sorted = np.take_along_axis(idx, order, axis=1)

        for bi, i in enumerate(range(s, e)):
            picked = []
            for j in idx_sorted[bi].tolist():
                if j == i:
                    continue
                if not bool(eligible[j]):
                    continue
                picked.append(j)
                if len(picked) >= topk:
                    break
            if len(picked) < topk:
                picked = picked + [-1] * (topk - len(picked))
            neigh[i] = np.array(picked, dtype=np.int64)
    return neigh


def _compose_one(
    item_idx: int,
    out_idx: int,
    item_title: str,
    item_brand: str,
    item_category: str,
    item_img_stem: str,
    neighbor_indices: list,
    neighbor_titles: list,
    neighbor_img_stems: list,
    images_dir: Path,
    out_dir: Path,
    width: int,
    height: int,
    padding: int,
    bg,
    text_color,
    title_font_size: int,
    small_font_size: int,
    collab_font_size: int,
    section_text: str,
    render_main_image: bool,
    render_main_text: bool,
    main_text_ratio: float,
    category_max_lines: int,
) -> bool:
    stem = str(item_img_stem) if item_img_stem is not None else ""
    base_path = images_dir / f"{stem}.jpg" if stem else images_dir / f"{int(item_idx)}.jpg"
    if bool(render_main_image):
        if not base_path.exists():
            base_path = images_dir / f"{int(item_idx)}.jpg"
            if not base_path.exists():
                return False

    try:
        canvas = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(canvas)
        font_title = _load_font(title_font_size)
        font_small = _load_font(small_font_size)
        font_collab = _load_font(collab_font_size)

        top_img_h = 0 if (not bool(render_main_image)) else int(height * 0.50)
        title_h = int(height * float(main_text_ratio))
        neigh_h = height - top_img_h - title_h
        if neigh_h < 1:
            neigh_h = 1

        if bool(render_main_image):
            with Image.open(base_path) as im:
                im = im.convert("RGB")
                w0, h0 = im.size
                scale = min((width - 2 * padding) / max(w0, 1), (top_img_h - 2 * padding) / max(h0, 1))
                nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
                im = im.resize((nw, nh), Image.BILINEAR)
                x0 = (width - nw) // 2
                y0 = padding
                canvas.paste(im, (x0, y0))

        tx0 = padding
        ty0 = top_img_h
        max_w = width - 2 * padding

        def _ellipsis_line(line: str, font: ImageFont.ImageFont, max_width: int) -> str:
            if draw.textlength(line, font=font) <= max_width:
                return line
            base = line
            while len(base) > 0 and draw.textlength(base + "...", font=font) > max_width:
                base = base[:-1]
            return (base + "...") if base else "..."

        title_lines = [ln for ln in _wrap_text_by_width(item_title, font_title, max_w, draw) if ln.strip()]
        if len(title_lines) > 3:
            title_lines = title_lines[:3]
            title_lines[-1] = _ellipsis_line(title_lines[-1], font_title, max_w)

        extra_lines = []
        b = (item_brand or "").strip()
        cat = (item_category or "").strip()
        if b:
            extra_lines.append("Brand: " + b)
        if cat:
            cat_prefix = "Category: " + cat
            cat_lines = [ln for ln in _wrap_text_by_width(cat_prefix, font_title, max_w, draw) if ln.strip()]
            if int(category_max_lines) > 0 and len(cat_lines) > int(category_max_lines):
                cat_lines = cat_lines[: int(category_max_lines)]
                cat_lines[-1] = _ellipsis_line(cat_lines[-1], font_title, max_w)
            extra_lines.extend(cat_lines)

        cur_y = ty0 + max(2, padding // 3)
        if bool(render_main_text):
            for ln in title_lines:
                bbox = draw.textbbox((0, 0), ln, font=font_title)
                lh = bbox[3] - bbox[1]
                if cur_y + lh > ty0 + title_h:
                    break
                draw.text((tx0, cur_y), ln, fill=text_color, font=font_title)
                cur_y += lh + 2
            for ln in extra_lines:
                ln = _ellipsis_line(ln, font_title, max_w)
                bbox = draw.textbbox((0, 0), ln, font=font_title)
                lh = bbox[3] - bbox[1]
                if cur_y + lh > ty0 + title_h:
                    break
                draw.text((tx0, cur_y), ln, fill=text_color, font=font_title)
                cur_y += lh + 2
        else:
            cur_y = ty0

        ny0 = cur_y + max(10, padding // 2)

        label = (section_text or "").strip()
        if label:
            lfont = font_title
            lb = draw.textbbox((0, 0), label, font=lfont)
            lh = lb[3] - lb[1]
            ly = ny0
            draw.text((tx0, ly), label, fill=text_color, font=lfont)
            ny0 = ly + lh + 2

        cols = len(neighbor_indices)
        if cols <= 0:
            cols = 3
        thumb_w = (width - padding * (cols + 1)) // cols
        avail_h = max(1, (height - padding) - ny0)
        reserve_for_titles = int(max(24, avail_h * 0.35))
        thumb_h = max(1, min(thumb_w, avail_h - reserve_for_titles))

        for j in range(min(cols, len(neighbor_indices))):
            ni = neighbor_indices[j]
            nt = neighbor_titles[j] if j < len(neighbor_titles) else ""
            x = padding + j * (thumb_w + padding)

            nstem = ""
            if j < len(neighbor_img_stems) and neighbor_img_stems[j] is not None:
                nstem = str(neighbor_img_stems[j])
            p = images_dir / f"{nstem}.jpg" if nstem else images_dir / f"{int(ni)}.jpg"
            if not p.exists():
                p = images_dir / f"{int(ni)}.jpg"
            if p.exists():
                with Image.open(p) as nim:
                    nim = nim.convert("RGB")
                    w1, h1 = nim.size
                    scale = min(thumb_w / max(w1, 1), thumb_h / max(h1, 1))
                    nw, nh = max(1, int(w1 * scale)), max(1, int(h1 * scale))
                    nim = nim.resize((nw, nh), Image.BILINEAR)
                    x_img = x + (thumb_w - nw) // 2
                    canvas.paste(nim, (x_img, ny0))

            t_y = ny0 + thumb_h + 2
            max_tw = thumb_w
            t_lines = [ln for ln in _wrap_text_by_width(nt, font_collab, max_tw, draw) if ln.strip()]
            if len(t_lines) > 3:
                t_lines = t_lines[:3]
                t_lines[-1] = _ellipsis_line(t_lines[-1], font_collab, max_tw)
            for ln in t_lines:
                bbox = draw.textbbox((0, 0), ln, font=font_collab)
                lh = bbox[3] - bbox[1]
                if t_y + lh > height - padding:
                    break
                draw.text((x, t_y), ln, fill=text_color, font=font_collab)
                t_y += lh + 1

        out_path = out_dir / f"{int(out_idx)}.jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, format="JPEG", quality=95, subsampling=2)
        return True
    except Exception:
        return False


def _compose_worker(job) -> bool:
    (
        item_idx,
        out_idx,
        item_title,
        item_brand,
        item_category,
        item_img_stem,
        neighbor_indices,
        neighbor_titles,
        neighbor_img_stems,
        images_dir,
        out_dir,
        width,
        height,
        padding,
        bg,
        text_color,
        title_font_size,
        small_font_size,
        collab_font_size,
        section_text,
        render_main_image,
        render_main_text,
        main_text_ratio,
        category_max_lines,
    ) = job
    return _compose_one(
        int(item_idx),
        int(out_idx),
        str(item_title),
        str(item_brand),
        str(item_category),
        str(item_img_stem),
        list(neighbor_indices),
        list(neighbor_titles),
        list(neighbor_img_stems),
        Path(images_dir),
        Path(out_dir),
        int(width),
        int(height),
        int(padding),
        tuple(bg),
        tuple(text_color),
        int(title_font_size),
        int(small_font_size),
        int(collab_font_size),
        str(section_text),
        bool(render_main_image),
        bool(render_main_text),
        float(main_text_ratio),
        int(category_max_lines),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_embedding", type=str, required=True, help="Path to item_embedding.npy")
    parser.add_argument("--item_mapping_csv", type=str, required=True, help="Path to item_id_mapping.csv")
    parser.add_argument("--meta_json", type=str, required=True, help="Metadata JSONL containing asin/title")
    parser.add_argument("--asin2id", type=str, required=True, help="Path to item_mapping.npy (asin->item_id)")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing raw item images named <item_id>.jpg")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for cards with collaboration space")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--candidate_k",
        type=int,
        default=50,
        help="Search this many top candidates per item before filtering for text-card availability.",
    )
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--backend", type=str, default="thread", choices=["thread", "process"])
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--title_font_size", type=int, default=20)
    parser.add_argument("--small_font_size", type=int, default=14)
    parser.add_argument("--collab_font_size", type=int, default=11, help="Font size for collaborator titles")
    parser.add_argument("--main_text_ratio", type=float, default=0.28)
    parser.add_argument("--category_max_lines", type=int, default=2)
    parser.add_argument(
        "--section_text",
        type=str,
        default="",
        help="Descriptive label shown between main and collaboration sections",
    )
    parser.add_argument(
        "--no_main_text",
        action="store_true",
        help="Disable rendering main item title/brand/category text (main image only).",
    )
    parser.add_argument(
        "--no_main_image",
        action="store_true",
        help="Disable rendering main item image (main text only).",
    )
    parser.add_argument(
        "--credit_text",
        type=str,
        default=None,
        help="(alias) Same as --section_text",
    )
    parser.add_argument("--only_missing", action="store_true")
    args = parser.parse_args()

    section_text = args.section_text
    if args.credit_text is not None:
        section_text = args.credit_text

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)

    emb = np.load(args.item_embedding)
    tokens = _read_mapping_csv(args.item_mapping_csv)

    if emb.shape[0] != len(tokens):
        raise ValueError(f"Embedding rows ({emb.shape[0]}) != mapping size ({len(tokens)})")

    n_items = len(tokens)
    titles_by_idx = [""] * n_items

    img_stem_by_idx = [""] * n_items
    for idx in range(1, n_items):
        t = tokens[idx]
        if t and str(t) != "[PAD]":
            img_stem_by_idx[idx] = str(t)

    asin2id = np.load(args.asin2id, allow_pickle=True).item()
    id2asin = {int(v): str(k) for k, v in asin2id.items()}
    asin2meta = {}
    for rec in _iter_jsonl(args.meta_json):
        a = rec.get("asin")
        if not a:
            continue
        asin2meta[str(a)] = rec

    def _idx_to_asin(idx: int):
        try:
            tok = tokens[idx]
        except Exception:
            tok = ""
        if tok and str(tok) in asin2meta:
            return str(tok)
        try:
            return id2asin.get(int(tok))
        except Exception:
            return None

    for idx in range(n_items):
        if idx == 0:
            continue
        asin = _idx_to_asin(idx)
        if asin:
            rec = asin2meta.get(str(asin), {})
            t = rec.get("title")
            if isinstance(t, str):
                titles_by_idx[idx] = t.strip()

    def _img_exists(idx: int) -> bool:
        stem = img_stem_by_idx[idx]
        if stem and (images_dir / f"{stem}.jpg").exists():
            return True
        return (images_dir / f"{idx}.jpg").exists()

    eligible = np.zeros((n_items,), dtype=bool)
    eligible_neighbor = np.zeros((n_items,), dtype=bool)
    for idx in range(1, n_items):
        if _img_exists(idx):
            eligible_neighbor[idx] = True
            eligible[idx] = True

    brands_by_idx = [""] * n_items
    cats_by_idx = [""] * n_items
    for idx in range(1, n_items):
        asin = _idx_to_asin(idx)
        if not asin:
            continue
        rec = asin2meta.get(str(asin), {})
        b = rec.get("brand")
        cats_val = rec.get("categories") or rec.get("category")
        if isinstance(b, str):
            brands_by_idx[idx] = b.strip()
        try:
            cats = cats_val
            if isinstance(cats, str):
                cats_by_idx[idx] = cats.strip()
            elif isinstance(cats, list) and len(cats) > 0:
                first = cats[0]
                if isinstance(first, list):
                    cats_flat = [str(c) for c in first][:4]
                else:
                    cats_flat = [str(c) for c in cats][:4]
                cats_by_idx[idx] = " / ".join(cats_flat)
        except Exception:
            pass

    neigh_idx = _filtered_topk_cosine_neighbors(
        emb,
        eligible=eligible_neighbor,
        topk=args.topk,
        candidate_k=args.candidate_k,
        block=args.block,
    )

    bg = (255, 255, 255)

    jobs = []
    for i in range(1, n_items):
        if not eligible[i]:
            continue
        tok = tokens[i] if i < len(tokens) else ""
        out_i = i
        try:
            if tok and str(tok) != "[PAD]":
                out_i = int(str(tok))
        except Exception:
            out_i = i
        out_path = out_dir / f"{out_i}.jpg"
        if args.only_missing and out_path.exists():
            continue
        nbrs = [j for j in neigh_idx[i].tolist() if j >= 0]
        nbr_titles = [titles_by_idx[j] for j in nbrs]
        nbr_img_stems = [img_stem_by_idx[j] for j in nbrs]
        jobs.append((
            i,
            out_i,
            titles_by_idx[i],
            brands_by_idx[i],
            cats_by_idx[i],
            img_stem_by_idx[i],
            nbrs,
            nbr_titles,
            nbr_img_stems,
        ))

    if len(jobs) == 0:
        raise RuntimeError(
            "No items to compose. This usually means no eligible items were found (need both image file in --images_dir and non-empty title from --meta_json). "
            "Double-check --images_dir/--meta_json/--asin2id and whether you passed --only_missing with an already-filled out_dir."
        )

    ok = 0
    bg = (255, 255, 255)
    text_color = (20, 20, 20)
    exec_jobs = [
        (
            int(i),
            int(out_i),
            str(t),
            str(b),
            str(cat),
            str(img_stem),
            list(nbrs),
            list(nbr_titles),
            list(nbr_img_stems),
            str(images_dir),
            str(out_dir),
            int(args.width),
            int(args.height),
            int(args.padding),
            bg,
            text_color,
            int(args.title_font_size),
            int(args.small_font_size),
            int(args.collab_font_size),
            str(section_text),
            (not bool(args.no_main_image)),
            (not bool(args.no_main_text)),
            float(args.main_text_ratio),
            int(args.category_max_lines),
        )
        for (i, out_i, t, b, cat, img_stem, nbrs, nbr_titles, nbr_img_stems) in jobs
    ]
    Executor = ThreadPoolExecutor if args.backend == "thread" else ProcessPoolExecutor
    with Executor(max_workers=max(1, args.workers)) as ex:
        fn = _compose_worker
        for r in tqdm(ex.map(fn, exec_jobs, chunksize=16 if args.backend == "process" else 64), total=len(exec_jobs), desc="Compose collab"):
            ok += 1 if r else 0

    np.save(out_dir / "topk_neighbors_idx.npy", neigh_idx)
    with open(out_dir / "topk_neighbors.csv", "w", encoding="utf-8") as f:
        f.write("item_index,item_token,neighbor_indices,neighbor_tokens\n")
        for i in range(1, n_items):
            nbrs = []
            for j in neigh_idx[i].tolist():
                if j < 0:
                    continue
                nbrs.append(j)
            nbr_tokens = [str(tokens[j]) for j in nbrs]
            f.write(f"{i},{tokens[i]},{'|'.join(map(str, nbrs))},{'|'.join(nbr_tokens)}\n")

    print(f"Done. composed={ok}/{len(jobs)} -> {out_dir}")


if __name__ == "__main__":
    main()
