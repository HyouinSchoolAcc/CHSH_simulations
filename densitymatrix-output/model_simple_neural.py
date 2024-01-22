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

def validate_model(model, loader):
    true_outputs = []
    predicted_outputs = []
    with torch.no_grad():
        for batch_inputs, batch_outputs in loader:
            predictions = model(batch_inputs.float()).flatten()
            predicted_outputs.extend(predictions.tolist())
            true_outputs.extend(batch_outputs.tolist())

    # Compute Spearman rank correlation coefficient
    correlation, _ = spearmanr(true_outputs, predicted_outputs)
    return correlation

# Best parameters from your Optuna study
best_hidden_dim = 128
best_lr = 0.0015010397675281019
best_epochs = 50

input_dim = 4 * 4 * 2
output_dim = 1

correlation = 0
retrain_count = 0
max_retrains = 10  # Setting a limit to avoid infinite retraining

while correlation < 0.99 and retrain_count < max_retrains:
    # Initialize and train the model with the best parameters
    model = SimpleNN(input_dim, best_hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

    for epoch in range(best_epochs):
        for batch_inputs, batch_outputs in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs.float())
            loss = criterion(predictions, batch_outputs.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

    # Validate the model
    correlation = validate_model(model, test_loader)
    print(f"Validation correlation: {correlation:.4f}")
    
    retrain_count += 1

# Save the trained model to a file
if correlation >= 0.99:
    torch.save(model.state_dict(), 'best_model.pth')
    print("Model saved with correlation >= 99%")
else:
    print("Max retrain attempts reached, could not achieve correlation >= 99%")