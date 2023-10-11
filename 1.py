import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(100, 64)  # Fully connected layer 1
        self.fc2 = nn.Linear(64, 32)  # Fully connected layer 2
        self.fc3 = nn.Linear(32, 5)   # Fully connected layer 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    '''
        def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Apply tanh activation
        x = torch.tanh(self.fc2(x))  # Apply tanh activation
        x = self.fc3(x)
        return x
    '''

# Create an instance of the model
model = MyModel()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Read data from file
with open("C:/Users/17536/Desktop/input.txt", "r") as file:
    lines = file.readlines()

# Prepare lists to store input and target data
input_data = []
target_data = []

# Process each line in the file
for line in lines:
    # Split the line by commas to extract input and target values
    values = line.strip().split(",")
    
    # Convert input values to floats and append to input_data
    input_values = [float(val) for val in values[5:]]
    input_data.append(input_values)
    
    # Convert target values to floats and append to target_data
    target_values = [float(val) for val in values[-5:]]
    target_data.append(target_values)

# Convert input_data and target_data to PyTorch tensors
input_data = torch.tensor(input_data)
target_data = torch.tensor(target_data)

print (input_data)
print (target_data)

# Train the model
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, input_data.size(0), batch_size):
        inputs = input_data[i:i+batch_size]
        targets = target_data[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print progress
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
