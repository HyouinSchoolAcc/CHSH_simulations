import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
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



class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

               # Convolutional layers
        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1).float()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1).float()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1,1), stride=(1,2))

        # Flatten and fully connected layers
        self.fc_input_size =  32*2*2
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.sigmoid = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)


        
    def forward(self, x):
        x=x.double()
        x = self.conv1(x.float())
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.relu4(x)
        x = self.fc4(x)
        
        return x
    
model = MyCNN().float()


# Create a DataLoader for batching and shuffling the data during training
batch_size = 8
data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
for batch in data_loader:
    input_batch, output_batch = batch  # Unpack batched input and output data

    #print("Batch input data:")
    #print(input_batch)
    #print("Batch output data:")
    #print(output_batch)

# Define your loss function (e.g., Mean Squared Error - MSE)
criterion = nn.MSELoss()

# Define your optimizer (e.g., Stochastic Gradient Descent - SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 4
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for inputs, labels in data_loader:
        # Convert inputs to double type
        inputs = inputs.float()
        labels = labels.float()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, labels.unsqueeze(1))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update the running loss
        running_loss += loss.item()
        
    # Print the average loss for the epoch
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}")



test_file_path = r"c:\Users\17536\Desktop\CHSH\data.txt"
test_dataset = MyDataset()

# Debug: Check the number of data samples
with open(test_file_path, 'r') as f:
    lines = f.readlines()
    num_samples2 = len(lines) // 17
print("Expected number of samples:", num_samples2)

with open(test_file_path, 'r') as f:
    for _ in range(num_samples2):
        matrix = np.zeros((4, 4, 2))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                line = f.readline().strip()
                row_data = [float(num) for num in line.split()]
                matrix[i, j, :] = row_data
        
        line = f.readline().strip()
        result = float(line)
        sample = {'input': matrix, 'output': result}
        test_dataset.data.append(sample)

# Debug: Check dataset size
print("Number of samples in test_dataset:", len(test_dataset.data))

# Create a DataLoader for the test data
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Prepare to collect predictions
all_predictions = []
all_labels = []  # Debug: Collect all labels

# Iterate through the test data and generate predictions
with torch.no_grad():
    for inputs, labels in test_data_loader:
        # Debug: Check DataLoader outputs
        print("Inputs shape:", inputs.shape)
        print("Labels shape:", labels.shape)
        
        inputs = inputs.float()
        outputs = model(inputs)
        predictions = outputs.numpy()
        
        all_predictions.append(predictions)
        all_labels.append(labels.numpy())  # Debug: Collect labels for each batch

# Concatenate all prediction batches and label batches
all_predictions = np.concatenate(all_predictions)
all_labels = np.concatenate(all_labels)  # Use this instead of the last labels batch

# Debug: Check predictions shape
print("Shape of all predictions:", all_predictions.shape)
print("Shape of all labels:", all_labels.shape)

all_predictions_flat = all_predictions.flatten()
labels_flat = all_labels.flatten()

print("Predictions:")
print(all_predictions_flat)

def rank_data(data):
    """Compute the ranks of the data."""
    order = data.argsort()
    ranks = order.argsort()
    return ranks + 1

def spearman_rank_correlation(data1, data2):
    """Compute Spearman's rank correlation coefficient."""
    assert len(data1) == len(data2), "Input lists must have the same length."
    
    rank1 = rank_data(data1)
    rank2 = rank_data(data2)
    
    d = rank1 - rank2
    d_squared = d**2
    
    n = len(data1)
    
    return 1 - (6 * sum(d_squared)) / (n * (n**2 - 1))

Spearman_correlation_transfer_frozen = spearman_rank_correlation(all_predictions_flat, labels_flat)
print("Spearman correlation of most frozen, with svr as ending = %.3f" % (Spearman_correlation_transfer_frozen))
