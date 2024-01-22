import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
class MyDataset:
    def __init__(self, file_path):
        self.data, self.targets = self._load_data(file_path)

    def _load_data(self, file_path):
        data, targets = [], []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(";")
                data.append([float(x) for x in parts[0].split(", ")])
                targets.append(float(parts[1]))
        return np.array(data), np.array(targets)

def train_and_evaluate(train_data, train_targets, test_data, test_targets):
    model = RandomForestRegressor()
    model.fit(train_data, train_targets)

    predictions = model.predict(test_data)
    mse = mean_squared_error(test_targets, predictions)
    print(f'Mean Squared Error: {mse}')

    # Compute Spearman's correlation
    spearman_corr, _ = spearmanr(test_targets, predictions)

    # Plot actual vs predicted values
    plt.scatter(test_targets, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values\nSpearman Correlation: {spearman_corr:.2f}')
    plt.show()

    return model

train_file_path = "data/measure_data.txt"
test_file_path = "data/measure_test.txt"

train_dataset = MyDataset(train_file_path)
test_dataset = MyDataset(test_file_path)

train_data, train_targets = train_dataset.data, train_dataset.targets
test_data, test_targets = test_dataset.data, test_dataset.targets

# Train the model and evaluate
model = train_and_evaluate(train_data, train_targets, test_data, test_targets)
