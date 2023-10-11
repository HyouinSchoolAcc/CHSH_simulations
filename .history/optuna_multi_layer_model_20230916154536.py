import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import optuna
from scipy.stats import spearmanr
import torch.optim as optim

class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self._load_data(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['input'], sample['output']

    def _load_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        num_samples = len(lines) // 17  # Each sample has 17 lines of data

        with open(file_path, 'r') as f:
            for _ in range(num_samples):
                matrix = np.zeros((4, 4, 2))
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        line = f.readline().strip()
                        row_data = [float(num) for num in line.split()]
                        matrix[i, j, :] = row_data

                # Read the result
                line = f.readline().strip()
                result = float(line)
                
                sample = {'input': matrix, 'output': result}
                self.data.append(sample)

# Creating instances for training and testing datasets
train_file_path = r"c:\Users\17536\Desktop\CHSH\data.txt"
test_file_path = r"c:\Users\17536\Desktop\CHSH\test.txt"

train_dataset = MyDataset(train_file_path)
test_dataset = MyDataset(test_file_path)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self, num_filters, kernel_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(2, num_filters, kernel_size=kernel_size, padding=1)
        self.fc1 = nn.Linear(num_filters * 4 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def objective(trial):
    # Define search space
    num_filters = trial.suggest_int('num_filters', 16, 64)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    
    model = CNNModel(num_filters, kernel_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(2):  # A very small number for demonstration
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.float()
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    total_loss = 0
    total_count = 0
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.float()
            labels = labels.float().unsqueeze(1)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_count += 1

            predictions.extend(outputs.flatten().tolist())
            ground_truths.extend(labels.flatten().tolist())

    rho, _ = spearmanr(predictions, ground_truths)
    
    return rho  # We want to maximize the Spearman correlation, hence negative sign

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)  # Number of trials can be increased for better optimization

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print(f'Value: {trial.value}')
    print(f'Params: {trial.params}')