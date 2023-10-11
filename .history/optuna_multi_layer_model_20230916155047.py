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

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=2)
        self.fc1 = nn.Linear(2*3*3, 1)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # Change the dimensions to (batch_size, channels, height, width)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Initialize the model, define the loss function and the optimizer
model = SimpleCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(60): # Adjust the number of epochs as necessary
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.float()
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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