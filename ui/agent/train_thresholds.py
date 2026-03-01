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
        
        # 过滤掉涨跌停板以上的无效跳空数据
        merged = merged[merged['True_Next_Return'].abs() < 0.21].copy()
                
    except Exception as e:
        print(f"   [错误] 处理预测数据异常: {e}")
            
    print(f"   [系统] 成功提取了 {len(merged)} 条真实双通道验证集交易样本！")
    return merged

def simulate_sharpe_ratio_fast(s_buy, buy, sell, s_sell, scores_sorted, y_true_sorted, same_symbol_mask, date_inverse, num_dates):
    if not (s_buy > buy and buy > sell and sell > s_sell):
        return -999.0 
    
    pos = np.full(len(scores_sorted), -1.0)
    pos[scores_sorted > s_sell] = -0.5
    pos[scores_sorted > sell] = 0.0
    pos[scores_sorted >= buy] = 0.5
    pos[scores_sorted >= s_buy] = 1.0
    
    prev_pos = np.roll(pos, 1)
    if len(prev_pos) > 0:
        prev_pos[0] = 0.0
    prev_pos[~same_symbol_mask] = 0.0
    
    COMMISSION = 0.0003
    SLIPPAGE   = 0.0002
    STAMP_TAX  = 0.0005
    
    pos_change = np.abs(pos - prev_pos)
    cost = pos_change * (COMMISSION + SLIPPAGE)
    reduce_mask = pos < prev_pos
    cost[reduce_mask] += np.abs(prev_pos[reduce_mask] - pos[reduce_mask]) * STAMP_TAX
    
    pnl = pos * y_true_sorted - cost
    
    daily_pnl_sum = np.bincount(date_inverse, weights=pnl, minlength=num_dates)
    daily_pnl_count = np.bincount(date_inverse, minlength=num_dates)
    daily_pnl = daily_pnl_sum / np.maximum(daily_pnl_count, 1)
    
    if len(daily_pnl) < 2:
        return 0.0
        
    avg = np.mean(daily_pnl)
    std = np.std(daily_pnl)
    if std == 0:
        return 0.0
        
    return float((avg / std) * math.sqrt(252))

def grid_search_thresholds(merged_df):
    print("🚀 启动端到端超参数回测网格搜索 (Numpy 终极加速版)...")
    import time
    
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
    
    df = merged_df.copy()
    df.sort_values(by=['Symbol', 'Date'], inplace=True)
    
    agent_scores = df['total_score'].values
    lstm_scores = np.clip(df['LSTM_Pred_Return'].values, -1.0, 1.0)
    y_true_sorted = df['True_Next_Return'].values
    
    symbols = df['Symbol'].values
    same_symbol_mask = np.ones(len(symbols), dtype=bool)
    if len(symbols) > 0:
        same_symbol_mask[0] = False
        same_symbol_mask[1:] = (symbols[1:] == symbols[:-1])
    
    dates = df['Date'].values
    unique_dates, date_inverse = np.unique(dates, return_inverse=True)
    num_dates = len(unique_dates)

    t0 = time.time()
    count = 0
    for k in k_range:
        w_ag = np.minimum(np.power(np.abs(agent_scores), k), 1.0)
        mask_07 = np.abs(agent_scores) > 0.7
        w_ag[mask_07] = np.maximum(w_ag[mask_07], 0.8)
        
        scores_sorted = (1.0 - w_ag) * lstm_scores + w_ag * agent_scores
        
        for sb in s_buy_range:
            for b in buy_range:
                for s in sell_range:
                    for ss in s_sell_range:
                        count += 1
                        if not (sb > b and b > s and s > ss):
                            continue
                        
                        sharpe = simulate_sharpe_ratio_fast(
                            sb, b, s, ss, scores_sorted, y_true_sorted, 
                            same_symbol_mask, date_inverse, num_dates
                        )
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_thresh = (sb, b, s, ss)
                            best_k = k
                            
                        if count % 5000 == 0:
                            print(f"   执行进度: {count} / {total_combinations} ...")
                            
    print(f"⏱️ 寻优总耗时: {time.time() - t0:.2f} 秒")
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
        if real_data.empty:
            print("未找到真实数据，退出循环...")
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
