'''import torch
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        #assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = MultilabelStratifiedKFold(n_splits=n_batches, shuffle=shuffle, random_state=0)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

        print('BatchSampler: x ', self.X.shape)
        print('BatchSampler: y ', self.y.shape)

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for _, test_idx in self.skf.split(self.X, self.y):
            #print_class_count(test_idx)
            print('yield: ', test_idx)
            yield test_idx

    def __len__(self):
        return len(self.y)'''