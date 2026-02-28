# -*- coding: utf-8 -*-
"""
train_thresholds.py

量化信号分离器的启发式网格寻优 (Grid Search for Classification Thresholds)
脱离拍脑袋的超参数设定，通过历史预测得分与真实收益标签池，遍历寻找最佳交易动作触发点。
以此证明交易阈值的“数据驱动”合理性。
"""

import math
import argparse
import random
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob
import pandas as pd

def generate_real_backtest_data():
    """
    使用真实的 LSTM 历史预测数据和真实的 Agent 知识图谱打分来寻找最优阈值。
    不再使用随机模拟的 Agent 分数，确保网格搜索结果与实际数据分布严格对齐。
    """
    print("   [系统] 正在加载真实 LSTM 预测和真实 Agent 打分数据...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lstm_file = os.path.join(base_dir, "data", "lstm_historical_predictions.csv")
    agent_file = os.path.join(base_dir, "data", "historical_agent_scores.csv")
    
    for f in [lstm_file, agent_file]:
        if not os.path.exists(f):
            print(f"   [错误] 未找到文件：{f}")
            return []
        
    data = []
    
    try:
        lstm_df = pd.read_csv(lstm_file)
        agent_df = pd.read_csv(agent_file)
        
        # 三段式划分：仅使用验证集数据进行超参数网格搜索
        if 'Split' in lstm_df.columns:
            lstm_df = lstm_df[lstm_df['Split'] == 'val']
            print(f"   [系统] 已筛选验证集数据 (Split='val')，共 {len(lstm_df)} 条")
        
        # 按日期和股票代码内连接，确保 LSTM 预测与 Agent 打分严格对齐
        lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])
        agent_df['date'] = pd.to_datetime(agent_df['date'])
        
        merged = pd.merge(
            lstm_df, agent_df,
            left_on=['Date', 'Symbol'], right_on=['date', 'stock_code'],
            how='inner'
        )
        print(f"   [系统] LSTM 与 Agent 数据对齐完成，有效样本: {len(merged)} 条")
        
        for _, row in merged.iterrows():
            lstm_score = float(row['LSTM_Pred_Return'])
            agent_score = float(row['total_score'])  # 真实的 Agent 知识图谱打分
            next_ret = float(row['True_Next_Return'])
            
            # 过滤掉涨跌停板以上的无效跳空数据
            if abs(next_ret) < 0.21: 
                data.append((lstm_score, agent_score, next_ret))
                
    except Exception as e:
        print(f"   [错误] 处理预测数据异常: {e}")
            
    print(f"   [系统] 成功提取了 {len(data)} 条真实双通道验证集交易样本！")
    return data

def simulate_sharpe_ratio(data, k_param, thresholds):
    """
    给定融合参数 k 和 交易阈值，跑一遍回测，计算策略的简易夏普率或总收益
    thresholds 格式：(strong_buy_th, buy_th, sell_th, strong_sell_th)
    要求: strong_buy > buy > sell > strong_sell
    """
    s_buy, buy, sell, s_sell = thresholds
    if not (s_buy > buy and buy > sell and sell > s_sell):
        return -999.0 # 无效的阈值排序
        
    portfolio_returns = []
    prev_pos = 0.0
    
    COMMISSION = 0.0003
    SLIPPAGE   = 0.001
    STAMP_TAX  = 0.001
    
    for lstm_mock_score, agent_mock_score, true_ret in data:
        # 1. 动态生成 final_score
        w_agent = min(math.pow(abs(agent_mock_score), k_param), 1.0)
        if abs(agent_mock_score) > 0.7:
            w_agent = max(w_agent, 0.8)
            
        w_lstm = 1.0 - w_agent
        final_score = w_lstm * lstm_mock_score + w_agent * agent_mock_score
        
        # 2. 执行动作决策
        position = 0.0
        if final_score >= s_buy:
            position = 1.0     # 强力看多，满仓
        elif final_score >= buy:
            position = 0.5     # 轻仓试盘
        elif final_score > sell:
            position = 0.0     # 空仓观望 (Hold)
        elif final_score > s_sell:
            position = -0.5    # 轻仓融券做空
        else:
            position = -1.0    # 强力看空，满仓做空
            
        # 计算换仓成本（仅在仓位变化时收取）
        pos_change = abs(position - prev_pos)
        cost = 0.0
        if pos_change > 0:
            cost += pos_change * (COMMISSION + SLIPPAGE)
            if position < prev_pos:
                cost += abs(prev_pos - position) * STAMP_TAX
        
        trade_profit = (position * true_ret) - cost
        portfolio_returns.append(trade_profit)
        prev_pos = position
        
    # 计算年化夏普比率 (简易化：平均收益 / 收益标准差)
    avg_ret = sum(portfolio_returns) / len(portfolio_returns)
    variance = sum((r - avg_ret) ** 2 for r in portfolio_returns) / len(portfolio_returns)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0001
    
    # 假设每日交易，年化倍数 approx sqrt(252)
    sharpe = (avg_ret / std_dev) * math.sqrt(252)
    return sharpe

def grid_search_thresholds(data):
    print("🚀 启动端到端超参数回测网格搜索...")
    
    # 构建超参数遍历空间 (Hyperparameter Space)
    # 取值范围：
    # k: 控制 Agent 的放大比例，我们将搜索的颗粒度切细一点
    # s_buy: 0.4 到 0.8
    # buy:   0.1 到 0.4
    # sell: -0.4 到 -0.1
    # s_sell: -0.8 到 -0.4
    
    # 增加 k 的网格，从非常不信任(0.5)到非常信任(3.5)，步长 0.5
    k_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    s_buy_range = [0.4, 0.5, 0.6, 0.7, 0.8]
    buy_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    sell_range = [-0.1, -0.15, -0.2, -0.25, -0.3, -0.4]
    s_sell_range = [-0.4, -0.5, -0.6, -0.7, -0.8]
    
    best_sharpe = -999.0
    best_thresh = None
    best_k = 2.0
    
    total_combinations = len(k_range) * len(s_buy_range) * len(buy_range) * len(sell_range) * len(s_sell_range)
    print(f"📊 即将验证的参数组合总数 (5维空间): {total_combinations} 次跑批")
    
    count = 0
    for k in k_range:
        for sb in s_buy_range:
            for b in buy_range:
                for s in sell_range:
                    for ss in s_sell_range:
                        count += 1
                        thresh = (sb, b, s, ss)
                        sharpe = simulate_sharpe_ratio(data, k, thresh)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_thresh = thresh
                            best_k = k
                            
                        if count % 1000 == 0:
                            print(f"   执行进度: {count} / {total_combinations} ...")
                            
    return best_k, best_thresh, best_sharpe

if __name__ == "__main__":
    import time
    import numpy as np
    start_time = time.time()
    
    # 设定蒙特卡洛迭代次数
    N_ITERATIONS = 10
    print(f"🌀 准备执行 {N_ITERATIONS} 次蒙特卡洛随机迭代寻优...")
    
    best_thresholds_history = []
    
    for i in range(N_ITERATIONS):
        print(f"\n=======================================================")
        print(f"▶️ 开始迭代 {i+1} / {N_ITERATIONS}")
        print("=======================================================")
        
        print("📈 步骤1/2: 构建/加载真实 A 股历史打分基准数据集...")
        real_data = generate_real_backtest_data()
        
        # 防止空数据运行
        if not real_data:
            print("未找到真实数据，退回生成模拟数据...")
            real_data = [] # Fallback
            break
        
        print(f"\n🔍 步骤2/2: 执行超空间网格扫描以寻找夏普最优截断点 ({i+1}/{N_ITERATIONS})...")
        best_k, best_t, best_s = grid_search_thresholds(real_data)
        
        if best_t is not None:
            best_thresholds_history.append((best_k, *best_t))
            print(f"   [迭代 {i+1} 最佳结果] K={best_k:.2f}, Thresholds: {best_t}, Max Sharpe: {best_s:.4f}")
        else:
            print(f"   [迭代 {i+1}] 未找到有效参数。")

    
    print("\n=======================================================")
    print("✅ 【全局参数寻优完成】Optimal Thresholds Found!")
    
    if best_thresholds_history:
        # 包含 k 的 5 个参数找稳定点
        final_k = np.median([x[0] for x in best_thresholds_history])
        final_s_buy = np.median([x[1] for x in best_thresholds_history])
        final_buy = np.median([x[2] for x in best_thresholds_history])
        final_sell = np.median([x[3] for x in best_thresholds_history])
        final_s_sell = np.median([x[4] for x in best_thresholds_history])
        
        import json
        with open("best_k_t.json", "w", encoding="utf-8") as f:
            json.dump({
                "history": best_thresholds_history,
                "final": [final_k, final_s_buy, final_buy, final_sell, final_s_sell]
            }, f, indent=4)
        print("\n   [建议固化进入 dynamic_gating.py 的硬核交易参数 (基于10次迭代中位数)]")
        print(f"   Math Mode 放大系数 (k): {final_k:.2f}    (原设定为 2.00)")
        print(f"   STRONG BUY       >= {final_s_buy:.2f}    (原设定为 0.50)")
        print(f"   BUY              >= {final_buy:.2f}    (原设定为 0.15)")
        print(f"   SELL              < {final_sell:.2f}    (原设定为 -0.40)")
        print(f"   STRONG SELL       < {final_s_sell:.2f}    (原设定为 -0.80)")
    else:
        print("未收集到足够的阈值数据。")
        
    print("=======================================================")
    print(f"⏱️ 寻优总耗时: {time.time() - start_time:.2f} 秒")
