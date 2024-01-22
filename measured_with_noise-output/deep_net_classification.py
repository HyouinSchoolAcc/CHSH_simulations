import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
import numpy as np
import optuna
import matplotlib.pyplot as plt
import os
# Dataset class
class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self._load_data(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['input'], sample['output'], sample['classification']

    def _load_data(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                data_part, output_part, classification_part = line.strip().split(';')
                input_data = np.array([float(num) for num in data_part.split(',')])
                output_value = float(output_part)
                classification_one_hot = np.array([int(num) for num in classification_part.split(',')])
                classification_label = np.argmax(classification_one_hot)  # Convert to class index
                print(classification_label)
                self.data.append({'input': input_data, 'output': output_value, 'classification': classification_label})

# Custom model class
class CustomModel(nn.Module):
    def __init__(self, shared_layers):
        super(CustomModel, self).__init__()
        self.shared_layers = nn.Sequential(*shared_layers)
        self.regressor = nn.Linear(shared_layers[-2].out_features, 1)  # Regression output
        self.classifier = nn.Linear(shared_layers[-2].out_features, 3)  # Classification output

    def forward(self, x):
        x = self.shared_layers(x)
        return self.regressor(x), self.classifier(x)

# Training and evaluation function
def train_and_evaluate(model, train_loader, test_loader, reg_criterion, class_criterion, optimizer, num_epochs=70):
    print("entered train and evaluate")
    classification_losses = []  # To record classification loss for each epoch
    
    for epoch in range(num_epochs):
        model.train()
        epoch_class_loss = 0.0
        for inputs, labels, classifications in train_loader:
            inputs = inputs.view(inputs.size(0), -1).float()
            labels = labels.float().unsqueeze(1)
            classifications = classifications.long()  # Ensure this is of Long type

            optimizer.zero_grad()
            reg_outputs, class_outputs = model(inputs)

            reg_loss = reg_criterion(reg_outputs, labels)
            class_loss = class_criterion(class_outputs, classifications)
            epoch_class_loss += class_loss.item()
            total_loss = reg_loss + class_loss
            total_loss.backward()
            optimizer.step()
        print("class loss is" + class_loss.item())
        classification_losses.append(epoch_class_loss / len(train_loader))
    print (f'Epoch {epoch+1}/{num_epochs}, Classification Loss: {classification_losses[-1]}')
    # Evaluation
    model.eval()
    true_outputs = []
    predicted_outputs = []
    with torch.no_grad():
        for inputs, labels, classifications in test_loader:
            inputs = inputs.view(inputs.size(0), -1).float()
            labels = labels.float().unsqueeze(1)

            reg_outputs, _ = model(inputs)
            predicted_outputs.extend(reg_outputs.flatten().tolist())
            true_outputs.extend(labels.flatten().tolist())

    # Compute Spearman rank correlation
    correlation, _ = spearmanr(true_outputs, predicted_outputs)
    return classification_losses, correlation

# Optuna objective function
def objective(trial):
    # Dynamic MLP architecture
    print('Trial Number:', trial.number)
    n_layers = trial.suggest_int('n_layers', 2, 5)
    layers = []
    in_features = 32
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    
    model = CustomModel(layers)



    reg_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-5, 1e-1, log=True))

    classification_losses, correlation = train_and_evaluate(model, train_loader, test_loader, reg_criterion, class_criterion, optimizer)
    print('Classification Losses:', classification_losses)
    # Plotting classification loss for each trial
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.figure()
    plt.plot(classification_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Classification Loss')
    plt.title(f'Classification Loss (Trial {trial.number})')
    plt.savefig(f'plots/classification_trial_{trial.number}.png')
    plt.close()

    return correlation

# Main execution
if __name__ == '__main__':
    train_file_path = "data/noise_data.txt"
    test_file_path = "data/noise_test.txt"

    train_dataset = MyDataset(train_file_path)
    test_dataset = MyDataset(test_file_path)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value:', trial.value)
    print('Params:', trial.params)
