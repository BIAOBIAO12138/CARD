import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
def process_data(file_path, mode, max_len, PAD_TOKEN=0):
    data = pd.read_parquet(file_path)

    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])

    if mode == 'train':
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append({
                    'history': sequence[:i],
                    'target': sequence[i]
                })
    elif mode == 'evaluation':
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            processed_data.append({
                'history': sequence[:-1],
                'target': sequence[-1]
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    if len(sequence) > max_len:
        return sequence[-max_len:]
    else:
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence
    
def item2code(code_path, codebook_size=256):
    data = np.load(code_path, allow_pickle=True)

    base, ext = os.path.splitext(code_path)
    ids_path = base + "_item_ids.npy"
    if os.path.exists(ids_path):
        item_ids = np.load(ids_path)
        if len(item_ids) != len(data):
            print(f"[WARN] item_ids length {len(item_ids)} != codes length {len(data)}; falling back to 1..N.")
            item_ids = np.arange(1, len(data) + 1, dtype=int)
    else:
        item_ids = np.arange(1, len(data) + 1, dtype=int)

    item_to_code = {}
    code_to_item = {}

    for code, iid in zip(data, item_ids):
        try:
            iid_int = int(iid)
        except (TypeError, ValueError):
            continue
        offsets = [c + i * codebook_size + 1 for i, c in enumerate(code)]
        item_to_code[iid_int] = offsets
        code_to_item[tuple(offsets)] = iid_int

    return item_to_code, code_to_item

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, mode, max_len, PAD_TOKEN=0):
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.item_to_code, self.code_to_item = item2code(code_path)
        self.data = self._prepare_data()
        
    def _prepare_data(self):
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
        )
        for item in processed_data:
            item['history'] = [self.item_to_code.get(x, np.array([self.PAD_TOKEN]*4)) for x in item['history']]
            item['target'] = self.item_to_code.get(item['target'], np.array([self.PAD_TOKEN]*4))
        return processed_data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
if __name__ == "__main__":
    dataset_path = '../data/Beauty/train.parquet'
    code_path = '../data/Beauty/Beauty_t5_rqvae.npy'
    mode = 'train'
    max_len = 20

    dataset = GenRecDataset(dataset_path, code_path, mode, max_len)
    print("Number of items in dataset:", len(dataset))

    print("First five items in dataset:", [dataset[i] for i in range(5)])
    
