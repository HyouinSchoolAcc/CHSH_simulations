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

class DynamicNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(DynamicNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective(trial):
    # Define constants
    INPUT_DIM = 32  # Replace with your input dimension
    OUTPUT_DIM = 1  # Replace with your output dimension
    EPOCHS = trial.suggest_int("epochs", 10, 50)
    LR = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Suggest number of layers
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # Suggest number of neurons for each layer
    hidden_layers = [trial.suggest_int(f"n_units_l{i}", 10, 100) for i in range(n_layers)]

    model = DynamicNN(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_layers=hidden_layers)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()  # Replace with your loss if it's not MSE

    # Training loop
    for epoch in range(EPOCHS):
        for data in train_loader:
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Validation or Test - Calculate Spearman correlation on test set
    predictions = []
    true_values = []

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            true_values.extend(targets.numpy())

    try:
        correlation, _ = spearmanr(predictions, true_values)
    except ValueError:
        correlation = -1  # or some penalty value if there's an error in correlation calculation

    return correlation


# Start the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")