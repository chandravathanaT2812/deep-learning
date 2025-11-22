import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, 
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib
import os

# 1. Seed Manager
class AndroSeedManager:
    @staticmethod
    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
AndroSeedManager.set_seed()

# 2. Custom Dataset Class for Time Series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 3. Custom LSTM RNN Time Series Model
class AndroLSTMRNNTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(AndroLSTMRNNTimeSeries, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take the output of the last time step

# 4. Custom Class for Training, Evaluation, and Inference
class AndroTimeSeriesPipeline:
    def __init__(self, model_params, seq_len=10):
        self.seq_len = seq_len
        self.model = AndroLSTMRNNTimeSeries(**model_params)
        self.scaler = MinMaxScaler()

    def preprocess_data(self, data):
        """Clean and scale data"""
        data = data.dropna()
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        return scaled_data

    def train(self, train_data, epochs=50, batch_size=32, lr=0.001):
        """Train the LSTM-RNN model"""
        train_loader = DataLoader(TimeSeriesDataset(train_data, self.seq_len), batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view(-1, self.seq_len, 1)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    def predict(self, data):
        """Make predictions using the trained model"""
        self.model.eval()
        test_loader = DataLoader(TimeSeriesDataset(data, self.seq_len), batch_size=1, shuffle=False)
        predictions = []
        with torch.no_grad():
            for x_batch, _ in test_loader:
                x_batch = x_batch.view(1, self.seq_len, 1)
                pred = self.model(x_batch).item()
                predictions.append(pred)
        return np.array(predictions)

    def save_model(self, model_path="andro_lstm_rnn_model_v2.pth", scaler_path="scaler_v2.pkl"):
        """Save the model and scaler"""
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        print("Model and scaler saved successfully.")

    def load_model(self, model_path="andro_lstm_rnn_model_v2.pth", scaler_path="scaler_v2.pkl"):
        """Load the pretrained model and scaler"""
        self.model.load_state_dict(torch.load(model_path))
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully.")

# 5. Usage Example with Extended Evaluation
if __name__ == "__main__":
    # Generate Sample Sales Data
    dates = pd.date_range(start="2023-01-01", periods=300)
    sales = np.sin(np.linspace(0, 50, 300)) * 50 + np.random.randn(300) * 5 + 100
    data = pd.DataFrame({"Date": dates, "Sales": sales})
    data.set_index("Date", inplace=True)

    # Hyperparameters
    model_params = {
        "input_size": 1,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2
    }

    # Initialize Pipeline
    pipeline = AndroTimeSeriesPipeline(model_params, seq_len=10)
    scaled_data = pipeline.preprocess_data(data)

    # Train Model
    pipeline.train(scaled_data, epochs=20, batch_size=16, lr=0.001)

    # Save Model and Scaler
    pipeline.save_model()

    # Reload Model
    pipeline.load_model()

    # Make Predictions
    predictions_loaded = pipeline.predict(scaled_data)
    actual = pipeline.scaler.inverse_transform(scaled_data[10:])
    predictions_rescaled = pipeline.scaler.inverse_transform(predictions_loaded.reshape(-1, 1))

    # Model Metrics
    mse = mean_squared_error(actual, predictions_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions_rescaled)
    r2 = r2_score(actual, predictions_rescaled)
    evs = explained_variance_score(actual, predictions_rescaled)

    print("\nModel Metrics After Loading:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Explained Variance: {evs:.4f}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[10:], actual, label="Actual Sales", color="blue")
    plt.plot(data.index[10:], predictions_rescaled, label="Predicted Sales", color="orange", linestyle="--")
    plt.title("Predictions vs Actual Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

    # Save Predictions in Multiple Formats
    submission_df = pd.DataFrame({"Date": data.index[10:], "Predicted_Sales": predictions_rescaled.flatten()})

    submission_df.to_csv("sales_predictions.csv", index=False)
    submission_df.to_excel("sales_predictions.xlsx", index=False)
    submission_df.to_json("sales_predictions.json", orient="records")
    submission_df.to_parquet("sales_predictions.parquet", engine="fastparquet", index=False)

    print("Predictions saved in CSV, Excel, JSON, and Parquet formats.")
