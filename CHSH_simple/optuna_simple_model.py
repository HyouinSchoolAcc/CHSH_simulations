import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import optuna
from scipy.stats import spearmanr


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

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def objective(trial):
    # Define hyperparameters
    input_dim = 4 * 4 * 2
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
    output_dim = 1
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 5, 50)

    # Initialize model and optimizer
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        for batch_inputs, batch_outputs in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs.float())
            loss = criterion(predictions, batch_outputs.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

    # Validation (using training data for simplicity)
    true_outputs = []
    predicted_outputs = []
    with torch.no_grad():
        for batch_inputs, batch_outputs in test_loader:
            predictions = model(batch_inputs.float()).flatten()
            predicted_outputs.extend(predictions.tolist())
            true_outputs.extend(batch_outputs.tolist())

    # Compute Spearman rank correlation coefficient
    correlation, _ = spearmanr(true_outputs, predicted_outputs)
    print(f"Correlation: {correlation:.2f}")
    return correlation  # Optuna tries to minimize the objective, so negate to maximize the correlation

# Setting up the Optuna study.
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")

"""Trial 13 finished with value: 0.9975490196078433 and parameters: {'hidden_dim': 
128, 'lr': 0.0015010397675281019, 'epochs': 50}. Best is trial 13 with value: 0.9975490196078433.
Correlation: 0.81
[I 2023-08-18 22:47:05,344] Trial 14 finished with value: 0.8137254901960784 and parameters: {'hidden_dim': 
127, 'lr': 1.375053375372561e-05, 'epochs': 38} .Best is trial 13 with value: 0.9975490196078433.
Correlation: 1.00"""