# -*- coding: utf-8 -*-
"""
生成 XGBoost 外部基线模型的预测结果 (作为外部基线对比)
读取已下载的股票量价CSV，使用 XGBoostRegressor 预测次日收益率，
并将盲测集的预测结果保存，以供 ablation_study.py 统一评测。
"""
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

base_dir = r"c:\Users\mortis\Desktop\ai4f\-LSTM-agent"
data_folder = os.path.join(base_dir, "ui", "stock_data_csv")

def add_technical_indicators(df):
    data = df.copy()
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    data['MA20'] = data['close'].rolling(window=20).mean()
    data['STD20'] = data['close'].rolling(window=20).std()
    data['BB_Upper'] = data['MA20'] + (data['STD20'] * 2)
    data['BB_Lower'] = data['MA20'] - (data['STD20'] * 2)
    return data

def prepare_xgb_data(file_path):
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df.sort_index(ascending=True, inplace=True)
    df.ffill(inplace=True)
    
    df['target_return'] = df['close'].pct_change().shift(-1) # 预测明天
    df = add_technical_indicators(df)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 滞后特征 (过去 5 天的收盘价涨跌幅)
    for i in range(1, 6):
        df[f'ret_lag_{i}'] = df['close'].pct_change().shift(i)
    df.dropna(inplace=True)
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower'] + \
                   [f'ret_lag_{i}' for i in range(1, 6)]
    
    return df[feature_cols].copy(), df['target_return'].copy()

print(f"🚀 开始训练 XGBoost 外部基准模型...")
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
xgb_predictions = []

for i, file_path in enumerate(csv_files, 1):
    base_name = os.path.basename(file_path).replace('.csv', '')
    parts = base_name.split('_')
    symbol = parts[-2] if len(parts) >= 2 else base_name
    
    print(f"[{i}/{len(csv_files)}] 训练 XGBoost: {symbol}")
    
    try:
        X_df, y_s = prepare_xgb_data(file_path)
        if len(X_df) < 200:
            continue
            
        X = X_df.values
        y = y_s.values
        dates = X_df.index
        
        # 必须与 LSTM 严格对齐样本划分比例！(前 70% 训练，中 10% 验证，后 20% 测试)
        train_size = int(len(X) * 0.7)
        val_end    = int(len(X) * 0.8)
        
        X_train, y_train = X[:val_end], y[:val_end] # XGBoost用前80%统称训练集
        X_test, y_test   = X[val_end:], y[val_end:]
        test_dates = dates[val_end:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, 
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train_s, y_train)
        
        preds = model.predict(X_test_s)
        
        for j in range(len(preds)):
            xgb_predictions.append({
                "Date": test_dates[j].strftime('%Y-%m-%d'),
                "Symbol": symbol,
                "XGB_Pred_Return": float(preds[j]),
                "True_Next_Return": float(y_test[j]) # 作为对齐校验
            })
            
    except Exception as e:
        print(f"  ❌ 失败: {e}")

xgb_df = pd.DataFrame(xgb_predictions)
out_path = os.path.join(base_dir, "data", "xgboost_historical_predictions.csv")
xgb_df.to_csv(out_path, index=False)
print(f"✅ XGBoost 盲测集预测已保存至: {out_path} (行数: {len(xgb_df)})")
