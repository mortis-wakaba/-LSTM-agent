# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
生成报告图表：
  图1 - 三种融合模型的累计净值曲线对比（盲测集）
  图2 - Math Gating 超参数 k 敏感性分析
"""
import sys, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

base_dir = r"c:\Users\mortis\Desktop\ai4f\-LSTM-agent"
data_dir = os.path.join(base_dir, "data")

# ============================================================
# 1. 加载数据
# ============================================================
lstm = pd.read_csv(os.path.join(data_dir, "lstm_historical_predictions.csv"))
agent = pd.read_csv(os.path.join(data_dir, "historical_agent_scores.csv"))
lstm['Date'] = pd.to_datetime(lstm['Date'])
agent['date'] = pd.to_datetime(agent['date'])

lstm_test = lstm[lstm['Split'] == 'test'].copy()
lstm_test['Symbol'] = lstm_test['Symbol'].astype(str)
agent['stock_code'] = agent['stock_code'].astype(str)
merged = pd.merge(lstm_test, agent, left_on=['Date', 'Symbol'], right_on=['date', 'stock_code'], how='inner')

# Cross-Attention 权重
import torch
sys.path.insert(0, os.path.join(base_dir, 'agent'))
from dynamic_gating import CrossAttentionGate
import ast

weights_path = os.path.join(data_dir, "cross_attention_weights.pth")
attn_model = CrossAttentionGate(lstm_hidden_dim=64, agent_dim=1)
attn_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
attn_model.eval()

hidden = np.array([ast.literal_eval(h) for h in merged['Hidden_State_64'].values])
all_hidden = torch.tensor(hidden, dtype=torch.float32)
all_agent_t = torch.tensor(merged['total_score'].values, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    weights = attn_model(all_hidden, all_agent_t)
    w_lstm_attn = weights[:, 0].cpu().numpy()
    w_agent_attn = weights[:, 1].cpu().numpy()

# ============================================================
# 2. 计算 3 种模型的预测分数（原始量纲，无归一化）
# ============================================================
lstm_scores = np.clip(merged['LSTM_Pred_Return'].values, -1.0, 1.0)
agent_scores = merged['total_score'].values
k_math = 3.1

models = {}

# Pure Agent
models['Pure Agent'] = agent_scores.copy()

# Math Gating
pred_math = np.zeros(len(merged))
for i in range(len(merged)):
    w_ag = min(math.pow(abs(agent_scores[i]), k_math), 1.0)
    if abs(agent_scores[i]) > 0.7:
        w_ag = max(w_ag, 0.8)
    pred_math[i] = (1.0 - w_ag) * lstm_scores[i] + w_ag * agent_scores[i]
models['Math Gating (k=3.1)'] = pred_math

# Cross-Attention
models['Cross-Attention (Ours)'] = w_lstm_attn * lstm_scores + w_agent_attn * agent_scores

# ============================================================
# 3. 模拟交易 → 逐日累计净值
# ============================================================
thresholds = (0.8, 0.1, -0.1, -0.4)
commission = 0.0003
stamp_tax = 0.0005
slippage = 0.0002

def get_daily_cumulative_nv(merged_df, pred_scores):
    t_strong_buy, t_buy, t_sell, t_strong_sell = thresholds
    records = []
    for idx in range(len(merged_df)):
        row = merged_df.iloc[idx]
        score = pred_scores[idx]
        true_ret = row['True_Next_Return']
        
        if score >= t_strong_buy:
            weight = 1.0
        elif score >= t_buy:
            weight = 0.5
        elif score <= t_strong_sell:
            weight = -1.0
        elif score <= t_sell:
            weight = -0.5
        else:
            weight = 0.0
        
        if weight != 0:
            cost = commission + slippage
            if weight < 0:
                cost += stamp_tax
            pnl = weight * true_ret - abs(weight) * cost
        else:
            pnl = 0.0
        
        records.append({'Date': row['Date'], 'PnL': pnl})
    
    df_pnl = pd.DataFrame(records)
    daily_pnl = df_pnl.groupby('Date')['PnL'].mean().sort_index()
    cum_nv = (1 + daily_pnl).cumprod()
    return daily_pnl.index.tolist(), cum_nv.values.tolist()

# ============================================================
# 4. 图1：三模型累计净值曲线
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

styles = {
    'Pure Agent':              ('#e67e22', '-.', 1.8),
    'Math Gating (k=3.1)':    ('#9b59b6', '--', 2.0),
    'Cross-Attention (Ours)':  ('#e74c3c', '-',  2.5),
}

for name, pred in models.items():
    dates, nv = get_daily_cumulative_nv(merged, pred)
    color, ls, lw = styles[name]
    ax.plot(dates, nv, color=color, linestyle=ls, linewidth=lw, 
            label=f'{name} (NV={nv[-1]:.4f})')

ax.axhline(y=1.0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xlabel('日期', fontsize=13)
ax.set_ylabel('累计净值', fontsize=13)
ax.set_title('盲测集：三种融合策略累计净值曲线对比', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.tight_layout()

chart1_path = os.path.join(data_dir, 'chart_cumulative_nv.png')
plt.savefig(chart1_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ 图1 累计净值折线图已保存: {chart1_path}")

# ============================================================
# 5. 图2：k 参数敏感性分析
# ============================================================
k_grid_path = os.path.join(data_dir, 'k_tuning_grid.csv')
if os.path.exists(k_grid_path):
    k_df = pd.read_csv(k_grid_path)
    
    fig2, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(k_df['k_value'], k_df['Sharpe'], 'o-', color='#e74c3c', 
             linewidth=2, markersize=5, label='Sharpe Ratio')
    best_idx = k_df['Sharpe'].idxmax()
    best_k = k_df.loc[best_idx, 'k_value']
    best_sharpe = k_df.loc[best_idx, 'Sharpe']
    ax1.plot(best_k, best_sharpe, '*', color='#e74c3c', markersize=18, zorder=10)
    ax1.annotate(f'最优 k={best_k:.1f}\nSharpe={best_sharpe:.3f}', 
                 xy=(best_k, best_sharpe), xytext=(best_k+0.3, best_sharpe-0.05),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    
    robust_mask = k_df['Sharpe'] >= best_sharpe * 0.95
    if robust_mask.any():
        k_robust = k_df.loc[robust_mask, 'k_value']
        ax1.axvspan(k_robust.min(), k_robust.max(), alpha=0.15, color='#e74c3c', 
                    label=f'稳健区间 k∈[{k_robust.min():.1f}, {k_robust.max():.1f}]')
    
    ax1.set_xlabel('超参数 k', fontsize=13)
    ax1.set_ylabel('Sharpe Ratio', color='#e74c3c', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax2 = ax1.twinx()
    ax2.plot(k_df['k_value'], k_df['Net_Value'], 's--', color='#27ae60', 
             linewidth=1.5, markersize=4, alpha=0.7, label='累计净值')
    ax2.set_ylabel('累计净值', color='#27ae60', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
    
    ax1.set_title('Math Gating 超参数 k 敏感性分析', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart2_path = os.path.join(data_dir, 'chart_k_sensitivity.png')
    plt.savefig(chart2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 图2 k敏感性分析图已保存: {chart2_path}")

print("✅ 全部图表生成完毕！")
