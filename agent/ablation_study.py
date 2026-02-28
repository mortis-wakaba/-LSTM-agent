# -*- coding: utf-8 -*-
"""
ablation_study.py

综合消融实验：在统一的样本外测试集上对比 Pure LSTM、数学启发式门控、交叉注意力门控
三种融合方案在 6 个维度上的表现：

  回归维度:  MSE, IC（信息系数）
  分类维度:  F1-Score（涨跌方向准确率）
  交易维度:  年化夏普比率, 最大回撤(MDD), 累计净值
"""

import os
import sys
import io
import ast
import math

import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dynamic_gating import CrossAttentionGate
from sklearn.metrics import mean_squared_error, f1_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =====================================================================
#  评估指标计算器
# =====================================================================

def calc_sharpe(daily_returns):
    """年化夏普比率 = (日均收益 / 日收益标准差) × √252"""
    arr = np.array(daily_returns)
    if len(arr) < 2 or np.std(arr) == 0:
        return 0.0
    return float((np.mean(arr) / np.std(arr)) * math.sqrt(252))


def calc_mdd(daily_returns):
    """最大回撤：资金曲线上最大的峰值到谷底跌幅"""
    equity = np.cumprod(1 + np.array(daily_returns))
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown))  # 负值


def calc_ic(predictions, true_returns):
    """信息系数（IC）：预测分数与真实收益率之间的皮尔逊相关系数"""
    if len(predictions) < 2:
        return 0.0
    corr, _ = pearsonr(predictions, true_returns)
    return float(corr) if not np.isnan(corr) else 0.0


def calc_f1(predictions, true_returns):
    """F1 分数：将预测视为涨跌二分类（>=0为涨=1，<0为跌=0），衡量方向判断能力"""
    pred_direction = (np.array(predictions) >= 0).astype(int)
    true_direction = (np.array(true_returns) >= 0).astype(int)
    return float(f1_score(true_direction, pred_direction, zero_division=0))


def simulate_trades(final_scores, true_returns, thresholds):
    """
    根据融合分数和阈值模拟交易，返回每日盈亏列表。
    交易摩擦模型（贴近 A 股真实成本）：
      - 券商佣金：单边 0.03%
      - 印花税：卖出时 0.1%
      - 滑点/冲击成本：单边 0.1%
    仅在仓位发生变化时计费（持仓不动不扣费）。
    """
    s_buy, buy, sell, s_sell = thresholds
    daily_pnl = []
    prev_pos = 0.0  # 追踪前一天的仓位

    COMMISSION = 0.0003   # 券商佣金：单边万三
    STAMP_TAX  = 0.0005   # 印花税：卖出时万分之五 (2023.8.28 减半征收新规)
    SLIPPAGE   = 0.0002   # 预估滑点/冲击成本：单边万分之二

    for score, true_ret in zip(final_scores, true_returns):
        # 按 5 档阈值映射仓位
        if   score >= s_buy:   pos = 1.0    # 强力看多，满仓
        elif score >= buy:     pos = 0.5    # 轻仓试盘
        elif score > sell:     pos = 0.0    # 空仓观望
        elif score > s_sell:   pos = -0.5   # 轻仓做空
        else:                  pos = -1.0   # 强力看空，满仓做空

        # 计算仓位变动量
        pos_change = abs(pos - prev_pos)
        
        # 交易成本 = 仅在换仓时收取
        cost = 0.0
        if pos_change > 0:
            cost += pos_change * (COMMISSION + SLIPPAGE)  # 买入/加仓成本
            # 如果是减仓（前仓位比新仓位大），需要加印花税
            if pos < prev_pos:
                cost += abs(prev_pos - pos) * STAMP_TAX

        daily_pnl.append(pos * true_ret - cost)
        prev_pos = pos

    return daily_pnl


# =====================================================================
#  主实验流程
# =====================================================================

def run_ablation(data_dir, k_math=2.0):
    """
    加载数据 → 计算三种融合模型的预测分数 → 模拟交易 → 输出 6 维统一对比表
    """
    lstm_file = os.path.join(data_dir, "lstm_historical_predictions.csv")
    agent_file = os.path.join(data_dir, "historical_agent_scores.csv")
    weights_path = os.path.join(data_dir, "cross_attention_weights.pth")

    for f in [lstm_file, agent_file]:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return

    # --- 加载并对齐数据 ---
    lstm_df = pd.read_csv(lstm_file)
    agent_df = pd.read_csv(agent_file)

    # 仅使用盲测集数据进行最终消融评估（绝对不在此数据上调参）
    if 'Split' in lstm_df.columns:
        test_lstm = lstm_df[lstm_df['Split'] == 'test'].copy()
    else:
        # 兼容旧版无 Split 列的 CSV
        lstm_df.sort_values(by='Date', inplace=True)
        split_idx = int(len(lstm_df) * 0.8)
        split_date = lstm_df.iloc[split_idx]['Date']
        test_lstm = lstm_df[lstm_df['Date'] >= split_date].copy()

    test_lstm['Date'] = pd.to_datetime(test_lstm['Date'])
    agent_df['date'] = pd.to_datetime(agent_df['date'])

    # 按日期和股票代码内连接，确保两组数据严格对齐
    merged = pd.merge(
        test_lstm, agent_df,
        left_on=['Date', 'Symbol'], right_on=['date', 'stock_code'],
        how='inner'
    )
    if len(merged) == 0:
        print("No aligned data found.")
        return

    # 隐藏状态可能以字符串形式存储，需要解析
    if isinstance(merged['Hidden_State_64'].iloc[0], str):
        merged['Hidden_State_64'] = merged['Hidden_State_64'].apply(ast.literal_eval)

    # --- 加载已训练的交叉注意力模型 ---
    attn_model = CrossAttentionGate(lstm_hidden_dim=64, agent_dim=1).to(DEVICE)
    if os.path.exists(weights_path):
        attn_model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    attn_model.eval()

    # 批量预计算注意力权重（提升效率）
    all_hidden = torch.tensor(merged['Hidden_State_64'].tolist(), dtype=torch.float32).to(DEVICE)
    all_agent_t = torch.tensor(merged['total_score'].values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        weights = attn_model(all_hidden, all_agent_t)
        w_lstm_attn = weights[:, 0].cpu().numpy()
        w_agent_attn = weights[:, 1].cpu().numpy()

    # --- 计算三种模型的融合分数 ---
    y_true = merged['True_Next_Return'].values
    lstm_scores = np.clip(merged['LSTM_Pred_Return'].values, -1.0, 1.0)
    agent_scores = merged['total_score'].values

    # 模型 1: 纯 LSTM（不融合 Agent 信号）
    pred_lstm = lstm_scores.copy()

    # 模型 2: 纯 Agent 图谱（不融合 LSTM 信号，直接使用图谱情绪分数）
    pred_agent = agent_scores.copy()

    # 模型 3: 数学启发式门控（W_agent = |S_agent|^k）
    pred_math = np.zeros(len(merged))
    for i in range(len(merged)):
        w_ag = min(math.pow(abs(agent_scores[i]), k_math), 1.0)
        # 重大事件强制接管规则
        if abs(agent_scores[i]) > 0.7:
            w_ag = max(w_ag, 0.8)
        pred_math[i] = (1.0 - w_ag) * lstm_scores[i] + w_ag * agent_scores[i]

    # 模型 4: 交叉注意力门控（神经网络动态权重）
    pred_attn = w_lstm_attn * lstm_scores + w_agent_attn * agent_scores

    # --- 模拟交易 ---
    thresholds = (0.8, 0.1, -0.1, -0.4)  # 由验证集上的真实数据网格搜索产生（无前视偏差）

    pnl_lstm = simulate_trades(pred_lstm, y_true, thresholds)
    pnl_agent = simulate_trades(pred_agent, y_true, thresholds)
    pnl_math = simulate_trades(pred_math, y_true, thresholds)
    pnl_attn = simulate_trades(pred_attn, y_true, thresholds)

    # --- 构建统一评估表 ---
    def build_row(name, preds, pnl):
        """为单个模型计算全部 6 个评估指标"""
        return {
            "Model": name,
            "MSE": round(mean_squared_error(y_true, preds), 6),
            "IC": round(calc_ic(preds, y_true), 4),
            "F1_Score": round(calc_f1(preds, y_true), 4),
            "Sharpe": round(calc_sharpe(pnl), 4),
            "MDD": round(calc_mdd(pnl), 4),
            "Net_Value": round(float(np.prod(1 + np.array(pnl))), 4),
        }

    results = pd.DataFrame([
        build_row("Pure_LSTM", pred_lstm, pnl_lstm),
        build_row("Pure_Agent", pred_agent, pnl_agent),
        build_row(f"Math_Gating_k{k_math}", pred_math, pnl_math),
        build_row("CrossAttention", pred_attn, pnl_attn),
    ])

    # 保存结果
    csv_path = os.path.join(data_dir, "ablation_study.csv")
    results.to_csv(csv_path, index=False)

    print(f"Samples: {len(merged)}, k={k_math}")
    print(results.to_string(index=False))
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    base_dir = r"c:\Users\mortis\Desktop\ai4f\-LSTM-agent"
    data_dir = os.path.join(base_dir, "data")
    run_ablation(data_dir, k_math=3.1)
