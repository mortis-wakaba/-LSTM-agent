# -*- coding: utf-8 -*-
"""
tune_k_parameter.py

专门针对 Math 门控中超参数 k 的高精度一维网格搜索（Grid Search）。
在固定最优离散交易水位线的前提下，地毯式遍历 k∈[0.1, 5.0]，
寻找能使【样本外测试集】年化夏普比率和累计净值最大化的黄金 k 值。
"""

import os
import sys
import io
import math
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calc_sharpe(daily_returns):
    """年化夏普比率"""
    arr = np.array(daily_returns)
    if len(arr) < 2 or np.std(arr) == 0:
        return 0.0
    return float((np.mean(arr) / np.std(arr)) * math.sqrt(252))

def calc_mdd(daily_returns):
    """最大回撤"""
    equity = np.cumprod(1 + np.array(daily_returns))
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown))

def search_best_k(data_dir):
    lstm_file = os.path.join(data_dir, "lstm_historical_predictions.csv")
    agent_file = os.path.join(data_dir, "historical_agent_scores.csv")

    print("Loading data...")
    lstm_df = pd.read_csv(lstm_file)
    agent_df = pd.read_csv(agent_file)

    # 仅使用验证集数据进行 k 值搜索
    if 'Split' in lstm_df.columns:
        test_lstm = lstm_df[lstm_df['Split'] == 'val'].copy()
    else:
        # 兼容旧版无 Split 列的 CSV
        lstm_df.sort_values(by='Date', inplace=True)
        split_idx = int(len(lstm_df) * 0.8)
        split_date = lstm_df.iloc[split_idx]['Date']
        test_lstm = lstm_df[lstm_df['Date'] >= split_date].copy()

    test_lstm['Date'] = pd.to_datetime(test_lstm['Date'])
    agent_df['date'] = pd.to_datetime(agent_df['date'])

    # 对齐数据
    merged = pd.merge(
        test_lstm, agent_df,
        left_on=['Date', 'Symbol'], right_on=['date', 'stock_code'],
        how='inner'
    )
    
    if len(merged) == 0:
        print("No aligned data found.")
        return

    print(f"Data aligned. Test Samples: {len(merged)}")
    
    y_true = merged['True_Next_Return'].values
    lstm_scores = np.clip(merged['LSTM_Pred_Return'].values, -1.0, 1.0)
    agent_scores = merged['total_score'].values

    # 固定最优的离散调仓阈值（由验证集上的真实数据网格搜索产生）
    thresholds = (0.8, 0.1, -0.1, -0.4)
    s_buy, buy, sell, s_sell = thresholds

    # 遍历空间：k = [0.1, 0.2, 0.3 ... 5.0]
    k_candidates = np.round(np.arange(0.1, 5.1, 0.1), 1)
    
    results = []

    COMMISSION = 0.0003   # 券商佣金：单边万三
    SLIPPAGE   = 0.001    # 滑点/冲击成本：单边千分之一
    STAMP_TAX  = 0.001    # 印花税：卖出时千分之一

    for k in k_candidates:
        pnl = []
        prev_pos = 0.0
        for i in range(len(merged)):
            # 融合公式
            w_ag = min(math.pow(abs(agent_scores[i]), k), 1.0)
            if abs(agent_scores[i]) > 0.7:  # 极端事件强制拉升比重
                w_ag = max(w_ag, 0.8)
            
            final_math = (1.0 - w_ag) * lstm_scores[i] + w_ag * agent_scores[i]

            # 阈值派单引擎
            if   final_math >= s_buy:   pos = 1.0
            elif final_math >= buy:     pos = 0.5
            elif final_math > sell:     pos = 0.0
            elif final_math > s_sell:   pos = -0.5
            else:                       pos = -1.0
            
            # 计算换仓成本
            pos_change = abs(pos - prev_pos)
            cost = 0.0
            if pos_change > 0:
                cost += pos_change * (COMMISSION + SLIPPAGE)
                if pos < prev_pos:
                    cost += abs(prev_pos - pos) * STAMP_TAX

            pnl.append(pos * y_true[i] - cost)
            prev_pos = pos

        # 计算业绩指标
        sharpe = calc_sharpe(pnl)
        mdd = calc_mdd(pnl)
        net_val = float(np.prod(1 + np.array(pnl)))
        
        # 将结果存入字典
        results.append({
            "k_value": k,
            "Sharpe": round(sharpe, 4),
            "Net_Value": round(net_val, 4),
            "MDD": round(mdd, 4)
        })

    # 转化为 DataFrame 并找出最优解
    df_res = pd.DataFrame(results)
    
    # 打印前 10 个最优的 K（按夏普率降序）
    print("\n=== Top 10 Best 'k' Parameters (Sorted by Sharpe) ===")
    top_10 = df_res.sort_values(by="Sharpe", ascending=False).head(10)
    print(top_10.to_string(index=False))

    best_k = df_res.iloc[df_res['Sharpe'].idxmax()]['k_value']
    print(f"\n✅ 结论: 对于当前的测试集数据分布，最优的超参数 k 为: {best_k}")
    
    # 保存全部网格日志
    csv_path = os.path.join(data_dir, "k_tuning_grid.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"完整网格搜索结果已保存至: {csv_path}")

if __name__ == "__main__":
    base_dir = r"c:\Users\mortis\Desktop\ai4f\-LSTM-agent"
    data_dir = os.path.join(base_dir, "data")
    search_best_k(data_dir)
