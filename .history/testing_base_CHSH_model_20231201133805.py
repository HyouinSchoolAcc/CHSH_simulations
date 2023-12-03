import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Assuming MyDataset and other necessary classes/functions are in the same file or imported
# from my_dataset import MyDataset  # Uncomment if MyDataset is in a separate file

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

# Define the model architecture (this should be the same as used in training)
class MyModel(nn.Module):
    def __init__(self, n_layers, layer_sizes):
        super(MyModel, self).__init__()
        layers = []
        in_features = 32  # Adjust according to your input features
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, layer_sizes[i]))
            layers.append(nn.ReLU())
            in_features = layer_sizes[i]
        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def load_model(model_path, n_layers, layer_sizes):
    layers = []
    in_features = 32  # Adjust according to your input features
    for i in range(n_layers):
        layers.append(nn.Linear(in_features, layer_sizes[i]))
        layers.append(nn.ReLU())
        in_features = layer_sizes[i]
    layers.append(nn.Linear(in_features, 1))

    model = nn.Sequential(*layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test_model(model, test_loader):
    # Evaluate the model
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.view(inputs.size(0), -1).float()
            outputs = model(inputs)
            print(outputs)
            # Add your evaluation logic here

if __name__ == '__main__':
    # Load the model
    model_path = r"c:\Users\17536\Desktop\CHSH\best_model_trial_1.pth" # Update with your model's path
    n_layers = 3  # Number of hidden layers
    layer_sizes = [17, 89, 49]
    model = load_model(model_path, n_layers, layer_sizes)

    # Load test data
    test_file_path = r"c:\Users\17536\Desktop\CHSH\test.txt"  # Update with your test data path
    test_dataset = MyDataset(test_file_path)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Test the model
    test_model(model, test_loader)