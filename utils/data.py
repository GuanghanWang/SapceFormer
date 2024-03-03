import os
import torch
import pandas as pd
import scanpy as sc
import scipy as sp
import squidpy as sq
import numpy as np
from utils.const import LABEL_DICT, CELLTYPE_NAME, X_COORDINATE_NAME, Y_COORDINATE_NAME


def load_data(args):
    data_list = []
    for file in os.listdir(args.data_dir):
        adata = sc.read_h5ad(os.path.join(args.data_dir, file))
        data_dict = {}

        data_dict['raw_X'] = torch.from_numpy(adata.X.toarray())

        # X
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=64)
        sc.pp.scale(adata, zero_center=False)
        highly_variable_genes = adata.X[:, adata.var['highly_variable'].to_numpy()].toarray()
        data_dict['highly_variable_genes'] = torch.from_numpy(highly_variable_genes)

        X = adata.X.toarray()
        data_dict['X'] = torch.from_numpy(X)

        # label
        labels = torch.tensor([LABEL_DICT[l] for l in adata.obs[CELLTYPE_NAME]])
        data_dict['labels'] = labels

        if args.spatial:
            adata.obsm['spatial'] = adata.obs[[X_COORDINATE_NAME, Y_COORDINATE_NAME]].to_numpy()
            data_dict['coordinates'] = torch.from_numpy(adata.obsm['spatial'])
            sq.gr.spatial_neighbors(adata, n_neighs=args.n_neighs)
            csr_matrix = adata.obsp['spatial_connectivities']
            rows, cols = csr_matrix.nonzero()
            
            values = np.ones_like(rows)
            edge_mask = sp.sparse.csr_matrix((values, (cols, rows)), shape=(adata.X.shape[0], adata.X.shape[0])).toarray()
            real_edge_mask = torch.from_numpy(edge_mask == 0)
            fake_edge_mask = torch.from_numpy(edge_mask == 1)

            data_dict['real_edge_mask'] = real_edge_mask
            data_dict['fake_edge_mask'] = fake_edge_mask

        data_list.append(data_dict)
    
    return data_list

