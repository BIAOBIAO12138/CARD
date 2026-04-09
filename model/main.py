import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import os
import sys
import random
import pandas as pd
from tqdm import tqdm
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import GenRecDataset
from model.dataloader import GenRecDataLoader
from model.prefix_tree import build_prefix_tree, make_prefix_allowed_tokens_fn

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'],
        vocab_size=config['vocab_size'],
        pad_token_id=config['pad_token_id'],
        eos_token_id=config['eos_token_id'],
        decoder_start_token_id=config['pad_token_id'],
        feed_forward_proj=config['feed_forward_proj'],
    )
        self.model = T5ForConditionalGeneration(t5config)
    
    @property
    def n_parameters(self):
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())
      emb_params = num_params(self.model.get_input_embeddings().parameters())
      return (
          f'Embedding parameters: {emb_params}\n'
          f'Non-embedding parameters: {total_params - emb_params}\n'
          f'Total trainable parameters: {total_params}\n'
      )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
      outputs = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
      )
      return outputs.loss, outputs.logits
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,  num_beams: int = 20, **kwargs):
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )

        gen_kwargs.update(kwargs)
        return self.model.generate(**gen_kwargs)

def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      for j in range(maxk):
        cur_pred = preds[i, j].tolist()
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

def recall_at_k(pos_index, k):
  return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
  ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
  dcg = 1.0 / torch.log2(ranks + 1)
  dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
  return dcg[:, :k].sum(dim=1).cpu().float()

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, topk_list, beam_size, device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--infer_size', type=int, default=256, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    parser.add_argument('--dataset_path', type=str, default='../data/Food', help='Path to the dataset')
    parser.add_argument('--code_path', type=str, default='../data/Food/Food_t5_rqvae.npy', help='Path to the item-to-code mapping file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation')
    parser.add_argument('--log_path', type=str, default='./logs/tiger.log', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./ckpt/tiger.pth', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=1, help='Run validation/test every N epochs')
    parser.add_argument('--topk_list', type=list, default=[5,10,20], help='List of top-k values for evaluation metrics')
    parser.add_argument('--beam_size', type=int, default=20, help='Beam size for generation')
    config = vars(parser.parse_args())
    os.makedirs(os.path.dirname(config['log_path']), exist_ok=True)
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")

    model = TIGER(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path']+ '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len']
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/valid.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len']
    )
    test_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/test.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len']
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    model.to(device)
    best_ndcg = 0.0
    
    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train(model, train_dataloader, optimizer, device)
        logging.info(f"Training loss: {train_loss}")

        if (epoch + 1) % config['eval_every'] != 0:
            continue

        avg_recalls, avg_ndcgs = evaluate(model, validation_dataloader, config['topk_list'], config['beam_size'], device)
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")
        test_avg_recalls, test_avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)
        logging.info(f"Test Dataset: {test_avg_recalls}")
        logging.info(f"Test Dataset: {test_avg_ndcgs}")

        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            logging.info(f"Best NDCG@20: {best_ndcg}")
            torch.save(model.state_dict(), config['save_path'])
            logging.info(f"Best model saved to {config['save_path']}")
        
