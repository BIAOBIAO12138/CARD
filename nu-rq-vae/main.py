import argparse
import random
import os
import sys
import torch
import numpy as np
import logging
import pandas as pd

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from rqvae4.datasets import EmbDataset
from nu_rqvae4.models.nu_rqvae import NURQVAE
from rqvae4.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="NU-RQ-VAE Index")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str, default="data/item_emb.parquet", help="Input data path.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256, 256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5, help='save limit for ckpt')
    parser.add_argument('--nvq_hidden_dim', type=int, default=64, help='hidden dim of NVQ transform h / h^{-1}')
    parser.add_argument('--nvq_loss_weight', type=float, default=1, help='weight of NVQ consistency loss')
    parser.add_argument('--nvq_nonlinearity', type=str, default='kumaraswamy', choices=['kumaraswamy', 'logistic'], help='NVQ nonlinearity')
    parser.add_argument('--export_dataset_nvq', action='store_true')
    parser.add_argument('--export_ckpt_path', type=str, default='')
    parser.add_argument('--export_ckpt_dir', type=str, default='')
    parser.add_argument('--export_out_parquet', type=str, default='')
    parser.add_argument('--export_id_col', type=str, default='ItemID')
    parser.add_argument('--export_batch_size', type=int, default=1024)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt/nu_rqvae4", help="output directory for model")
    return parser.parse_args()


def _resolve_ckpt_path(args):
    if args.export_ckpt_path:
        return args.export_ckpt_path
    if args.export_ckpt_dir:
        cand = os.path.join(args.export_ckpt_dir, 'best_collision_model.pth')
        if os.path.exists(cand):
            return cand
        cand = os.path.join(args.export_ckpt_dir, 'best_loss_model.pth')
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f'Checkpoint not found in dir: {args.export_ckpt_dir}')
    raise ValueError('Please provide --export_ckpt_path or --export_ckpt_dir when --export_dataset_nvq is set')


@torch.no_grad()
def export_dataset_nvq(args):
    device = torch.device(args.device)
    ckpt_path = _resolve_ckpt_path(args)

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    ckpt_args = ckpt['args']
    state_dict = ckpt['state_dict']

    df = pd.read_parquet(args.data_path)
    if 'embedding' not in df.columns:
        raise KeyError(f"Cannot find column 'embedding' in {args.data_path}. Columns={list(df.columns)}")

    emb = df['embedding'].values
    emb = np.stack(emb, axis=0).astype(np.float32)

    ids = None
    if args.export_id_col and args.export_id_col in df.columns:
        ids = df[args.export_id_col].to_numpy()

    model = NURQVAE(
        in_dim=emb.shape[-1],
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        beta=ckpt_args.beta,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
        nvq_hidden_dim=getattr(ckpt_args, 'nvq_hidden_dim', ckpt_args.e_dim),
        nvq_loss_weight=getattr(ckpt_args, 'nvq_loss_weight', 1.0),
        nvq_nonlinearity=getattr(ckpt_args, 'nvq_nonlinearity', 'kumaraswamy'),
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    out_parquet = args.export_out_parquet
    if not out_parquet:
        base = os.path.splitext(os.path.basename(args.data_path))[0]
        out_parquet = os.path.join(os.path.dirname(args.data_path) or '.', f'{base}_z_prime.parquet')
    os.makedirs(os.path.dirname(out_parquet) or '.', exist_ok=True)

    z_prime_all = np.empty((emb.shape[0], model.e_dim), dtype=np.float32)
    bs = int(args.export_batch_size)
    for start in tqdm(range(0, emb.shape[0], bs), desc='export_z_prime'):
        end = min(start + bs, emb.shape[0])
        x = torch.from_numpy(emb[start:end]).to(device)
        z = model.encoder(x)
        z_prime = model._nvq_h(z)
        z_prime_all[start:end] = z_prime.detach().cpu().numpy().astype(np.float32)

    df_out = pd.DataFrame({'embedding': [row for row in z_prime_all]})
    if ids is not None:
        df_out[args.export_id_col] = ids
        cols = [args.export_id_col, 'embedding']
        df_out = df_out[cols]

    df_out.to_parquet(out_parquet, index=False)
    print('Saved:', out_parquet)


if __name__ == '__main__':
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    if args.export_dataset_nvq:
        export_dataset_nvq(args)
        sys.exit(0)

    data = EmbDataset(args.data_path)
    model = NURQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        nvq_hidden_dim=args.nvq_hidden_dim,
        nvq_loss_weight=args.nvq_loss_weight,
        nvq_nonlinearity=args.nvq_nonlinearity,
    )
    print(model)

    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)
