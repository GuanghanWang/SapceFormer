from torch.utils.data import Dataset


class BrainDataset(Dataset):
    def __init__(self, data_list, spatial):
        self.data = data_list
        self.spatial = spatial

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        X = sample['X']
        labels = sample['labels']
        raw_X = sample['raw_X']
        highly_variable_genes = sample['highly_variable_genes']
        if self.spatial:
            coordinates = sample['coordinates']
            real_edge_mask = sample['real_edge_mask']
            fake_edge_mask = sample['fake_edge_mask']
            return X, labels, real_edge_mask, fake_edge_mask, coordinates, highly_variable_genes, raw_X
        else:
            return X, labels, raw_X