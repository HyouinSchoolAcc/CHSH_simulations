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
        num_samples = len(lines) 

        with open(file_path, 'r') as f:
            for _ in range(num_samples):
                line = f.readline().strip()
                # Split the line at the semicolon
                data_part, chsh_part = line.split(";")

                # Process the 20-element array
                data_array = np.array([float(num) for num in data_part.split(", ")])

                # Process the CHSH value
                chsh_value = float(chsh_part.strip())

                # Create a sample dictionary
                sample = {'input': data_array, 'output': chsh_value}
                
                print("Type of inputs:", type(data_array))

                self.data.append(sample)
# Creating instances for training and testing datasets
train_file_path = r"data\measure_data.txt"
test_file_path = r"data\measure_test.txt"

train_dataset = MyDataset(train_file_path)
test_dataset = MyDataset(test_file_path)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

def objective(trial):
    # Dynamic MLP architecture
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    in_features = 24  # Adjust this according to your input features
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, 1))
    
    model = nn.Sequential(*layers)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform('lr', 1e-5, 1e-1))

    # Training loop
    for epoch in range(70):
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.view(inputs.size(0), -1).float()  # Flatten the inputs
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Verification step
    true_outputs = []
    predicted_outputs = []
    with torch.no_grad():
        for batch_inputs, batch_outputs in test_loader:
            batch_inputs = batch_inputs.view(batch_inputs.size(0), -1).float()
            predictions = model(batch_inputs).flatten()
            predicted_outputs.extend(predictions.tolist())
            true_outputs.extend(batch_outputs.tolist())

    # Compute Spearman rank correlation coefficient
    correlation, _ = spearmanr(true_outputs, predicted_outputs)

    if correlation > 0.95:
        # Save the best model
        torch.save(model.state_dict(), f'save_models/measure_models/best_model_trial_{trial.number}.pth')

    return correlation

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ', trial.params)