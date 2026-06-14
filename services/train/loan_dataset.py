import torch
from torch.utils.data import Dataset


class loan_dataset(Dataset):
    def __init__(self, dataset, train=True):
        self.train = train
        target = "loan_paid_back"

        # train acts a submission flag
        if self.train:
            # Generate Labels
            X = dataset.drop(columns=[target]).values.astype("float32")
            self.features = torch.tensor(X, dtype=torch.float32)
            Y = dataset[target].values.astype("float32")
            self.labels = torch.tensor(Y, dtype=torch.float32)
        else:
            X = dataset.values.astype("float32")
            self.features = torch.tensor(X, dtype=torch.float32)
            self.labels = None  # No labels for inference

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.train:
            return self.features[index], self.labels[index]
        else:
            return self.features[index]
