import argparse
import os
import os.path as op
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def _expand_history_target(df: pd.DataFrame, include_history: bool) -> List[Tuple[int, int, int]]:
    if "history" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Expected columns ['history','target'], got: {list(df.columns)}")

    triples: List[Tuple[int, int, int]] = []
    for uid, row in enumerate(df.itertuples(index=False)):
        hist = getattr(row, "history")
        tgt = getattr(row, "target")

        if isinstance(hist, np.ndarray):
            hist = hist.tolist()

        if not isinstance(hist, (list, tuple)):
            raise ValueError(f"history must be list/tuple, got {type(hist)}")

        hist_items = [int(x) for x in hist if x is not None]
        if include_history:
            for t, iid in enumerate(hist_items, start=1):
                triples.append((int(uid), int(iid), int(t)))
            triples.append((int(uid), int(tgt), int(len(hist_items) + 1)))
        else:
            triples.append((int(uid), int(tgt), int(len(hist_items) + 1)))

    return triples


def _to_inter_df(triples: Iterable[Tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(triples, columns=["user_id:token", "item_id:token", "timestamp:float"])


def _read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet: {path}.\n"
            "You need pyarrow or fastparquet in the current python env.\n"
            f"Original error: {e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TIGER parquet splits to RecBole .inter benchmark files")
    parser.add_argument("--dataset_name", type=str, default="Food")
    parser.add_argument("--parquet_dir", type=str, required=True, help="Directory containing train/valid/test.parquet")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save RecBole *.inter files. If not set, defaults to <repo_root>/dataset/<dataset_name>.",
    )
    parser.add_argument(
        "--leave_two_out",
        action="store_true",
        help="Use leave-two-out semantics: train=history only, valid=train.target (2nd last), test=test.target (last).",
    )
    parser.add_argument(
        "--target_only_for_valid_test",
        action="store_true",
        help="(Legacy) Write only target interactions for valid/test using their own targets.",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        repo_root = op.dirname(op.dirname(op.abspath(__file__)))
        args.out_dir = op.join(repo_root, "dataset", str(args.dataset_name))

    os.makedirs(args.out_dir, exist_ok=True)

    split_paths = {
        "train": op.join(args.parquet_dir, "train.parquet"),
        "valid": op.join(args.parquet_dir, "valid.parquet"),
        "test": op.join(args.parquet_dir, "test.parquet"),
    }

    for k, p in split_paths.items():
        if not op.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")

    if args.leave_two_out:
        df_train = _read_parquet(split_paths["train"])
        df_test = _read_parquet(split_paths["test"])
        if len(df_train) != len(df_test):
            raise ValueError(
                f"leave_two_out requires same number of rows/users in train and test, got train={len(df_train)} test={len(df_test)}"
            )

        train_triples: List[Tuple[int, int, int]] = []
        for uid, row in enumerate(df_train.itertuples(index=False)):
            hist = getattr(row, "history")
            if isinstance(hist, np.ndarray):
                hist = hist.tolist()
            if not isinstance(hist, (list, tuple)):
                raise ValueError(f"history must be list/tuple, got {type(hist)}")
            hist_items = [int(x) for x in hist if x is not None]
            for t, iid in enumerate(hist_items, start=1):
                train_triples.append((int(uid), int(iid), int(t)))
        train_df = _to_inter_df(train_triples)
        out_path = op.join(args.out_dir, f"{args.dataset_name}.train.inter")
        train_df.to_csv(out_path, sep="\t", index=False, header=True)
        print(f"[OK] Saved train interactions: {len(train_df)} -> {out_path}")

        valid_triples: List[Tuple[int, int, int]] = []
        for uid, row in enumerate(df_train.itertuples(index=False)):
            hist = getattr(row, "history")
            tgt = getattr(row, "target")
            if isinstance(hist, np.ndarray):
                hist = hist.tolist()
            hist_len = len([x for x in hist if x is not None]) if isinstance(hist, (list, tuple)) else 0
            valid_triples.append((int(uid), int(tgt), int(hist_len + 1)))
        valid_df = _to_inter_df(valid_triples)
        out_path = op.join(args.out_dir, f"{args.dataset_name}.valid.inter")
        valid_df.to_csv(out_path, sep="\t", index=False, header=True)
        print(f"[OK] Saved valid interactions: {len(valid_df)} -> {out_path}")

        test_triples: List[Tuple[int, int, int]] = []
        for uid, row in enumerate(df_test.itertuples(index=False)):
            hist = getattr(row, "history")
            tgt = getattr(row, "target")
            if isinstance(hist, np.ndarray):
                hist = hist.tolist()
            hist_len = len([x for x in hist if x is not None]) if isinstance(hist, (list, tuple)) else 0
            test_triples.append((int(uid), int(tgt), int(hist_len + 1)))
        test_df = _to_inter_df(test_triples)
        out_path = op.join(args.out_dir, f"{args.dataset_name}.test.inter")
        test_df.to_csv(out_path, sep="\t", index=False, header=True)
        print(f"[OK] Saved test interactions: {len(test_df)} -> {out_path}")
        return

    for split in ["train", "valid", "test"]:
        df = _read_parquet(split_paths[split])
        include_history = True
        if args.target_only_for_valid_test and split in {"valid", "test"}:
            include_history = False
        triples = _expand_history_target(df, include_history=include_history)
        inter_df = _to_inter_df(triples)

        out_path = op.join(args.out_dir, f"{args.dataset_name}.{split}.inter")
        inter_df.to_csv(out_path, sep="\t", index=False, header=True)
        print(f"[OK] Saved {split} interactions: {len(inter_df)} -> {out_path}")


if __name__ == "__main__":
    main()
