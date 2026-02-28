import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import ast

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dynamic_gating import CrossAttentionGate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FusionDataset(Dataset):
    def __init__(self, lstm_df, agent_df):
        # Merge on Date and Symbol
        lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])
        agent_df['date'] = pd.to_datetime(agent_df['date'])
        
        # Format names
        merged = pd.merge(
            lstm_df, agent_df, 
            left_on=['Date', 'Symbol'], 
            right_on=['date', 'stock_code'], 
            how='inner'
        )
        
        # Some hidden states might be saved as strings
        if isinstance(merged['Hidden_State_64'].iloc[0], str):
            merged['Hidden_State_64'] = merged['Hidden_State_64'].apply(ast.literal_eval)
            
        self.lstm_hs = torch.tensor(merged['Hidden_State_64'].tolist(), dtype=torch.float32)
        # Agent score is 1D
        self.agent_scores = torch.tensor(merged['total_score'].values, dtype=torch.float32).unsqueeze(1)
        
        # We need to predict the true next return
        self.true_returns = torch.tensor(merged['True_Next_Return'].values, dtype=torch.float32).unsqueeze(1)
        # And we need the baseline LSTM prediction
        self.lstm_preds = torch.tensor(merged['LSTM_Pred_Return'].values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.true_returns)
        
    def __getitem__(self, idx):
        return self.lstm_hs[idx], self.agent_scores[idx], self.lstm_preds[idx], self.true_returns[idx]

def train_fusion_model(data_dir, epochs=20, lr=0.001):
    lstm_file = os.path.join(data_dir, "lstm_historical_predictions.csv")
    agent_file = os.path.join(data_dir, "historical_agent_scores.csv")
    
    if not os.path.exists(lstm_file) or not os.path.exists(agent_file):
        print("Data files not ready yet. Please ensure both LSTM and Agent scripts have finished.")
        return
        
    print("Loading datasets...")
    lstm_df = pd.read_csv(lstm_file)
    agent_df = pd.read_csv(agent_file)
    
    # Sort by date
    lstm_df.sort_values(by='Date', inplace=True)
    
    # 仅使用验证集数据训练交叉注意力模型
    if 'Split' in lstm_df.columns:
        val_data = lstm_df[lstm_df['Split'] == 'val'].copy()
    else:
        # 兼容旧版无 Split 列的 CSV
        split_idx = int(len(lstm_df) * 0.8)
        split_date = lstm_df.iloc[split_idx]['Date']
        val_data = lstm_df[lstm_df['Date'] >= split_date].copy()
    
    # 在验证集内部再切 80/20，用于交叉注意力的训练和早停
    sub_split = int(len(val_data) * 0.8)
    train_lstm = val_data.iloc[:sub_split]
    val_lstm   = val_data.iloc[sub_split:]
    
    print(f"Creating PyTorch DataLoaders (using validation split)...")
    train_dataset = FusionDataset(train_lstm, agent_df)
    val_dataset = FusionDataset(val_lstm, agent_df)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    model = CrossAttentionGate(lstm_hidden_dim=64, agent_dim=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # L2 reg
    criterion = nn.SmoothL1Loss() # Huber Loss
    
    best_val_loss = float('inf')
    output_weights_path = os.path.join(data_dir, "cross_attention_weights.pth")
    
    print("\nStarting Training CrossAttentionGate...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for lstm_h, agent_score, lstm_pred, true_ret in train_loader:
            lstm_h, agent_score = lstm_h.to(DEVICE), agent_score.to(DEVICE)
            lstm_pred, true_ret = lstm_pred.to(DEVICE), true_ret.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass to get attention weights [batch, 2]
            weights = model(lstm_h, agent_score)
            w_lstm = weights[:, 0].unsqueeze(1)
            w_agent = weights[:, 1].unsqueeze(1)
            
            # The final gated output is a weighted sum! 
            # Note: The original agent score is NOT a return percentage, it's an impact score [-1, 1].
            # For the loss calculation, we assume the agent score is scaled to map to potential returns,
            # or we let the network learn how to fuse them. Let's create a learnable scaling for the agent score.
            
            # Since the objective is just to minimize MSE between the fused output and true return,
            # We construct a linear head that projects the weighted combination into a return prediction.
            # But wait, the standard Fusion Engine doesn't have a linear head. It just uses mathematical fusion.
            # Let's align with Fusion Engine:
            # final_score = w_lstm * lstm_score + w_agent * agent_score
            # The final_score represents the trading confidence, not the exact return rate.
            # But we CAN train it by matching final_score -> true return (scaled).
            # To avoid adding more layers outside CrossAttentionGate, we let CrossAttentionGate 
            # output the weights that minimize the distance between (w_lstm*lstm_pred + w_agent*(agent_score*alpha)) and true return.
            
            # 使用可学习的量纲映射系数 alpha（由网络自适应优化）
            fused_pred = w_lstm * lstm_pred + w_agent * (agent_score * model.alpha)
            
            loss = criterion(fused_pred, true_ret)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(true_ret)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lstm_h, agent_score, lstm_pred, true_ret in val_loader:
                lstm_h, agent_score = lstm_h.to(DEVICE), agent_score.to(DEVICE)
                lstm_pred, true_ret = lstm_pred.to(DEVICE), true_ret.to(DEVICE)
                
                weights = model(lstm_h, agent_score)
                w_lstm = weights[:, 0].unsqueeze(1)
                w_agent = weights[:, 1].unsqueeze(1)
                
                fused_pred = w_lstm * lstm_pred + w_agent * (agent_score * model.alpha)
                loss = criterion(fused_pred, true_ret)
                val_loss += loss.item() * len(true_ret)
                
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_weights_path)
            
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} *")
        
    print(f"\nTraining Complete. Best weights saved to: {output_weights_path}")

if __name__ == "__main__":
    base_dir = r"c:\Users\mortis\Desktop\ai4f\-LSTM-agent"
    data_dir = os.path.join(base_dir, "data")
    train_fusion_model(data_dir)
