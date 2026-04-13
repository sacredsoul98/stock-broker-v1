import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import settings
from utils.logger import get_logger

logger = get_logger("models.train_lstm")

class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(PriceLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.dropout_layer(out[:, -1, :])
        out = self.fc(out) 
        return out

def train_lstm_model(df, X_seq, y_val, model_path=os.path.join(settings.MODEL_DIR, "lstm_model.pth"), scaler_path=os.path.join(settings.MODEL_DIR, "scaler_lstm.pkl")):
    """
    Trains PyTorch LSTM model. X_seq shape: (samples, timesteps, features)
    """
    logger.info("Initializing LSTM Deep Learning Training...")
    
    # Time-based split (80% train, 20% test)
    split_idx = int(len(X_seq) * 0.8)
    
    X_train_seq = X_seq[:split_idx]
    y_train = y_val[:split_idx]
    X_test_seq = X_seq[split_idx:]
    y_test = y_val[split_idx:]
    
    # Define test indices corresponding to the original dataframe
    test_indices = df.index[settings.LSTM_SEQ_LENGTH + split_idx:] # Account for warmup
    
    samples, timesteps, features = X_train_seq.shape
    
    # Flatten for scaling, then reshape back
    scaler_X = StandardScaler()
    X_train_flat = X_train_seq.reshape(-1, features)
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(samples, timesteps, features)
    
    samples_test = X_test_seq.shape[0]
    X_test_flat = X_test_seq.reshape(-1, features)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(samples_test, timesteps, features)
    
    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.from_numpy(X_train_scaled).type(torch.Tensor)
    y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor).view(-1, 1)
    
    x_test_tensor = torch.from_numpy(X_test_scaled).type(torch.Tensor)
    y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor).view(-1, 1)
    
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=settings.LSTM_BATCH_SIZE, shuffle=False)
    
    model = PriceLSTM(
        input_dim=features, 
        hidden_dim=settings.LSTM_HIDDEN_DIM, 
        num_layers=settings.LSTM_NUM_LAYERS,
        dropout=getattr(settings, 'LSTM_DROPOUT', 0.2)
    )
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = settings.LSTM_EPOCHS
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(x_test_tensor).numpy()
    
    mse = np.mean((y_test.reshape(-1, 1) - test_preds)**2)
    rmse = np.sqrt(mse)
    
    logger.info("--- LSTM Test Metrics ---")
    logger.info(f"MSE:  {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_X, scaler_path)
    logger.info(f"LSTM model saved to {model_path} and scaler to {scaler_path}")
    
    # Results mapped to test set index
    results = pd.DataFrame(index=test_indices)
    results['LSTM_Pred_Price'] = test_preds.flatten()
    
    return results, model, scaler_X
