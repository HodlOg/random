class Dataset:
    def __init__(self, X, y):
        self.X = X  # Features
        self.y = y  # Labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]