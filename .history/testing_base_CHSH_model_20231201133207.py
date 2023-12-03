import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Assuming MyDataset and other necessary classes/functions are in the same file or imported
# from my_dataset import MyDataset  # Uncomment if MyDataset is in a separate file

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
    model = MyModel(n_layers, layer_sizes)
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
            # Add your evaluation logic here

if __name__ == '__main__':
    # Load the model
    model_path = r"c:\Users\17536\Desktop\CHSH\best_model_trial_1.pth" # Update with your model's path
    n_layers = 3  # Number of layers used in the model, adjust accordingly
    layer_sizes = [64, 32, 16]  # Sizes of layers used in the model, adjust accordingly
    model = load_model(model_path, n_layers, layer_sizes)

    # Load test data
    test_file_path = r"c:\Users\17536\Desktop\CHSH\test.txt"  # Update with your test data path
    test_dataset = MyDataset(test_file_path)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Test the model
    test_model(model, test_loader)