import torch
import math
import numpy as np
import scanpy as sc
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr


class LocalAttention(nn.Module):
    def __init__(self, d_model):
        super(LocalAttention, self).__init__()
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.d_model)
        scores.masked_fill_(mask, -1e9)
        max_values, _ = torch.max(scores, dim=1, keepdim=True)
        scores = torch.exp(scores - max_values)

        attn = scores / torch.sum(scores, dim=-1, keepdim=True)
        cntx = torch.matmul(attn, V)
    
        return cntx, attn


class LocalTransformerCelltype(nn.Module):
    def __init__(self, dropout, input_dim, ffn_dim, hidden_dim, output_dim):
        super(LocalTransformerCelltype, self).__init__()
        self.encoder = LocalAttention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = LocalAttention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x, real_edge_mask, fake_edge_mask):
        hidden, encode_weights = self.encoder(x, real_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        return h


class SpatialAttention(nn.Module):
    def __init__(self, d_model, gamma):
        super(SpatialAttention, self).__init__()
        self.gamma = gamma
        self.d_model = d_model

        self.Q_real = nn.Linear(d_model, d_model, bias=False)
        self.Q_fake = nn.Linear(d_model, d_model, bias=False)
        self.K_real = nn.Linear(d_model, d_model, bias=False)
        self.K_fake = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, real_edge_mask, fake_edge_mask):
        Q_real = self.Q_real(x)
        K_real = self.K_real(x)
        Q_fake = self.Q_fake(x)
        K_fake = self.K_fake(x)
        V = self.V(x)

        real_scores = torch.matmul(Q_real, K_real.transpose(0, 1)) / math.sqrt(self.d_model)
        real_scores.masked_fill_(real_edge_mask, -1e9)
        real_scores_max, _ = torch.max(real_scores, dim=1, keepdim=True)
        fake_scores = torch.matmul(Q_fake, K_fake.transpose(0, 1)) / math.sqrt(self.d_model)
        fake_scores.masked_fill_(fake_edge_mask, -1e9)
        fake_scores_max, _ = torch.max(fake_scores, dim=1, keepdim=True)
        max_scores = torch.maximum(real_scores_max, fake_scores_max)

        real_scores = real_scores - max_scores
        real_scores = torch.exp(real_scores) / (1 + self.gamma)
        fake_scores = fake_scores - max_scores
        fake_scores = self.gamma * torch.exp(fake_scores) / (1 + self.gamma)

        scores = real_scores + fake_scores

        attn = scores / torch.sum(scores, dim=-1, keepdim=True)
        cntx = torch.matmul(attn, V)
    
        return cntx, attn


class SpaceFormerCelltype(nn.Module):
    def __init__(self, dropout, input_dim, ffn_dim, hidden_dim, output_dim, gamma):
        super(SpaceFormerCelltype, self).__init__()
        self.encoder = SpatialAttention(input_dim, gamma)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = SpatialAttention(input_dim, gamma)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, real_edge_mask, fake_edge_mask):
        hidden, encode_weights = self.encoder(x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        return h


def build_model_celltype(args):
    if args.model == 'LocalTransformer':
        model = LocalTransformerCelltype(args.dropout, args.input_dim, args.ffn_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'SpaceFormer':
        model = SpaceFormerCelltype(args.dropout, args.input_dim, args.ffn_dim, args.hidden_dim, args.output_dim, args.gamma)
    return model
    

class LocalTransformerPretrain(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim):
        super(LocalTransformerPretrain, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = LocalAttention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = LocalAttention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.ff1 = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def encoding_mask_nodes(self, x, cell_mask_rate, gene_mask_rate):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(cell_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        mask_x = x[mask_nodes].clone()
        gene_mask = torch.zeros_like(mask_x, dtype=torch.bool).to(x.device)
        for i in range(mask_x.size(0)):
            num_to_zero =  int(gene_mask_rate * mask_x.size(1))
            indices_to_zero = torch.randperm(mask_x.size(1))[:num_to_zero]
            mask_x[i, indices_to_zero] = 0
            gene_mask[i, indices_to_zero] = True
        
        out_x = x.clone()
        out_x[mask_nodes] = mask_x

        return out_x, mask_nodes, gene_mask

    def forward(self, x, real_edge_mask, fake_edge_mask):
        use_x, mask_nodes, gene_mask = self.encoding_mask_nodes(x, self.cell_mask_rate, self.gene_mask_rate)

        hidden, encode_weights = self.encoder(use_x, real_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + use_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        hidden[mask_nodes][gene_mask] = 0

        recon, decode_weights = self.decoder(hidden, real_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        recon = self.norm4(recon + self.ff1(recon))

        h = self.head(recon)

        x_init = x[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)
        x_recon = h[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)

        return x_init, x_recon, encode_weights, recon


class SpaceFormerPretrain(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim, gamma):
        super(SpaceFormerPretrain, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = SpatialAttention(input_dim, gamma)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = SpatialAttention(input_dim, gamma)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.ff1 = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def encoding_mask_nodes(self, x, cell_mask_rate, gene_mask_rate):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(cell_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        mask_x = x[mask_nodes].clone()
        gene_mask = torch.zeros_like(mask_x, dtype=torch.bool).to(x.device)
        for i in range(mask_x.size(0)):
            num_to_zero =  int(gene_mask_rate * mask_x.size(1))
            indices_to_zero = torch.randperm(mask_x.size(1))[:num_to_zero]
            mask_x[i, indices_to_zero] = 0
            gene_mask[i, indices_to_zero] = True
        
        out_x = x.clone()
        out_x[mask_nodes] = mask_x

        return out_x, mask_nodes, gene_mask

    def forward(self, x, real_edge_mask, fake_edge_mask):
        use_x, mask_nodes, gene_mask = self.encoding_mask_nodes(x, self.cell_mask_rate, self.gene_mask_rate)

        hidden, encode_weights = self.encoder(use_x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + use_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        hidden[mask_nodes][gene_mask] = 0

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        recon = self.norm4(recon + self.ff1(recon))

        h = self.head(recon)

        x_init = x[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)
        x_recon = h[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)

        return x_init, x_recon, encode_weights, recon


def build_model_pretrain(args):
    if args.model == 'LocalTransformer':
        model = LocalTransformerPretrain(args.cell_mask_rate, args.gene_mask_rate, args.dropout, args.input_dim, args.ffn_dim)
    elif args.model == 'SpaceFormer':
        model = SpaceFormerPretrain(args.cell_mask_rate, args.gene_mask_rate, args.dropout, args.input_dim, args.ffn_dim, args.gamma)
    return model
    

class LocalTransformerImputation(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout, imputation_rate):
        super(LocalTransformerImputation, self).__init__()

        self.imputation_rate = imputation_rate

        self.encoder = LocalAttention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = LocalAttention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def drop(self, x, gene_list, num_drop):
        mask = torch.zeros_like(x, dtype=bool)
    
        for i in range(x.shape[0]):
            random_indices = torch.randperm(len(gene_list))[:num_drop]
            dropout_indices = gene_list[random_indices]
            mask[i, dropout_indices] = True
        
        drop_x = x.clone()
        drop_x[mask] = 0

        return drop_x, mask

    def forward(self, x, gene_list, real_edge_mask, fake_edge_mask):
        drop_x, drop_mask = self.drop(x, gene_list, int(self.imputation_rate * len(gene_list)))
        
        hidden, encode_weights = self.encoder(drop_x, real_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        x_init = x[drop_mask].reshape(x.shape[0], -1)
        x_recon = h[drop_mask].reshape(x.shape[0], -1)

        return x_init, x_recon

    def preprocess(self, x):
        adata = sc.AnnData(X=x.detach().cpu().numpy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        X = adata.X
        return torch.from_numpy(X).to(x.device)
    
    def evaluation(self, raw, gene_list, real_edge_mask, fake_edge_mask):
        drop_raw, drop_mask = self.drop(raw, gene_list, int(self.imputation_rate * len(gene_list)))
        drop_x = self.preprocess(drop_raw)

        hidden, encode_weights = self.encoder(drop_x, real_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)
        
        pearson_list = []
        spearman_list = []
        for i in range(len(gene_list)):
            gene_idx = gene_list[i]

            gene_mask_i = drop_mask[:, gene_idx]
            
            gene_init = raw[:, gene_idx][gene_mask_i]
            gene_recon = h[:, gene_idx][gene_mask_i]

            pearson, _ = pearsonr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            pearson_list.append(pearson)
            spearman, _ = spearmanr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            spearman_list.append(spearman)

        return pearson_list, spearman_list


class SpaceFormerImputation(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout, imputation_rate, gamma):
        super(SpaceFormerImputation, self).__init__()

        self.imputation_rate = imputation_rate

        self.encoder = SpatialAttention(input_dim, gamma)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = SpatialAttention(input_dim, gamma)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def drop(self, x, gene_list, num_drop):
        mask = torch.zeros_like(x, dtype=bool)
    
        for i in range(x.shape[0]):
            random_indices = torch.randperm(len(gene_list))[:num_drop]
            dropout_indices = gene_list[random_indices]
            mask[i, dropout_indices] = True
        
        drop_x = x.clone()
        drop_x[mask] = 0

        return drop_x, mask

    def forward(self, x, gene_list, real_edge_mask, fake_edge_mask):
        drop_x, drop_mask = self.drop(x, gene_list, int(self.imputation_rate * len(gene_list)))
        
        hidden, encode_weights = self.encoder(drop_x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        x_init = x[drop_mask].reshape(x.shape[0], -1)
        x_recon = h[drop_mask].reshape(x.shape[0], -1)

        return x_init, x_recon
    
    def preprocess(self, x):
        adata = sc.AnnData(X=x.detach().cpu().numpy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        X = adata.X
        return torch.from_numpy(X).to(x.device)
    
    def evaluation(self, raw, gene_list, real_edge_mask, fake_edge_mask):
        drop_raw, drop_mask = self.drop(raw, gene_list, int(self.imputation_rate * len(gene_list)))
        drop_x = self.preprocess(drop_raw)

        hidden, encode_weights = self.encoder(drop_x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)
        
        pearson_list = []
        spearman_list = []
        for i in range(len(gene_list)):
            gene_idx = gene_list[i]

            gene_mask_i = drop_mask[:, gene_idx]
            
            gene_init = raw[:, gene_idx][gene_mask_i]
            gene_recon = h[:, gene_idx][gene_mask_i]

            pearson, _ = pearsonr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            pearson_list.append(pearson)
            spearman, _ = spearmanr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            spearman_list.append(spearman)

        return pearson_list, spearman_list


def build_model_imputation(args):
    if args.model == 'LocalTransformer':
        model = LocalTransformerImputation(args.input_dim, args.ffn_dim, args.dropout, args.imputation_rate)
    elif args.model == 'SpaceFormer':
        model = SpaceFormerImputation(args.input_dim, args.ffn_dim, args.dropout, args.imputation_rate, args.gamma)
    return model