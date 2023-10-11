import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import optuna
from scipy.stats import spearmanr


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['input'], sample['output']
    
file_path = r"c:\Users\17536\Desktop\CHSH\data.txt"
my_dataset = MyDataset()

with open(file_path, 'r') as f:
    lines = f.readlines()
num_samples = len(lines) // 17 # Each sample has 17 lines of data
print("Number of samples:", num_samples) 
# Prepare lists to store input and target data

with open(file_path, 'r') as f:
    for i in range(num_samples):
        matrix = np.zeros((4, 4, 2))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                line = f.readline().strip()
                row_data = [float(num) for num in line.split()]
                matrix[i, j, :] = row_data

    # Read the result
        line = f.readline().strip()
        result = float(line)

        #print("Matrix:")
        #print(matrix[:,:,0])
        #print(matrix[:,:,1])
        #print("Result:", result)

        sample = {'input': matrix, 'output': result}
        my_dataset.data.append(sample)

    if num_samples * (17) == len(lines):
        print(f"Data successfully read ({num_samples} records).")
    else:
        print("Error: Incomplete data.")


data_loader = DataLoader(my_dataset, batch_size=8, shuffle=True)

class MyCNN(nn.Module):
    def __init__(self, num_conv_layers, num_fc_layers, num_kernels, fc_units):
        super(MyCNN, self).__init__()

        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        input_channel = 4
        for i in range(num_conv_layers):
            self.convs.append(nn.Conv2d(input_channel, num_kernels[i], kernel_size=3, stride=1, padding=1).float())
            self.relus.append(nn.ReLU())
            self.pools.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))) 
            input_channel = num_kernels[i]
        
        self.fcs = nn.ModuleList()
        input_size = input_channel * 4 * 2  # Assuming a size of 4x4x2 matrix
        for units in fc_units:
            self.fcs.append(nn.Linear(input_size, units))
            input_size = units
        self.fcs.append(nn.Linear(input_size, 1))

    def forward(self, x):
        for conv, relu, pool in zip(self.convs, self.relus, self.pools):
            x = pool(relu(conv(x.float())))
        
        x = x.view(x.size(0), -1)  # Flatten
        for fc in self.fcs[:-1]:
            x = torch.relu(fc(x))
        x = self.fcs[-1](x)
        
        return x

 
def objective(trial):
    # Hyperparameters to optimize
    batch_size = trial.suggest_int('batch_size', 4, 64)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int('num_epochs', 1, 10)
    
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    num_kernels = [int(trial.suggest_discrete_uniform(f'num_kernels_{i}', 16, 128, 16)) for i in range(num_conv_layers)]

    num_fc_layers = trial.suggest_int('num_fc_layers', 1, 3)
    fc_units = [int(trial.suggest_discrete_uniform(f'fc_units_{i}', 32, 512, 32)) for i in range(num_fc_layers)]

    model = MyCNN(num_conv_layers, num_fc_layers, num_kernels, fc_units).float()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    all_labels = []
    all_predictions = []

    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            inputs = inputs.float()
            labels = labels.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            # Store labels and predictions for correlation computation
            all_labels.extend(labels.tolist())
            all_predictions.extend(outputs.squeeze(1).detach().tolist())

    # Calculate the Spearman correlation
    correlation, _ = spearmanr(all_labels, all_predictions)

    # Return the negative of the correlation to maximize it
    return -correlation

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(" Value: ", trial.value)
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")