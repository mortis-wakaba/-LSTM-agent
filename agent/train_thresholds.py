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
    使用真实的 A 股历史 K 线数据构建回测集。
    我们依靠过去的动量特征（模拟 LSTM）和随机情绪（模拟 Agent 新闻）来合成当时的系统打分，
    并使用明天的真实涨跌幅 (next_day_pct_change) 作为真正的奖励反馈。
    """
    print("   [系统] 正在加载所有 50 只股票的真实历史 K 线数据...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "ui", "stock_data_csv")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print("   [错误] 在 ui/stock_data_csv/ 未找到数据！")
        return []
        
    data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) < 10:
                continue
                
            # 找到明天的真实涨跌幅作为基准（也就是我们今天收盘后预测，明天去赚的钱）
            # pct_change 是百分比 (比如 3.5 代表 3.5%)，我们转为小数 0.035
            df['next_ret'] = df['pct_change'].shift(-1) / 100.0
            
            # 使用近 5 日累计收益率的平滑值来替代 LSTM 模型基准分
            df['momentum_5d'] = df['pct_change'].rolling(5).mean() / 100.0
            
            df = df.dropna()
            
            for _, row in df.iterrows():
                # 模拟系统预测打分：
                # 我们假设系统对均值回归/动量有一定的捕捉能力（这里简化为动量 + 一点噪音）
                # 真实情况 LSTM 会输出一个 [-1.0, 1.0] 的置信度
                lstm_mock_score = min(max(row['momentum_5d'] * 5, -1.0), 1.0)
                
                # Agent 新闻是稀疏的，偶尔发生大偏差
                agent_mock_score = 0.0
                if random.random() < 0.1: # 10%的概率有突发新闻
                    agent_mock_score = random.uniform(-1.0, 1.0)
                    
                # 按照 Fusion Engine 逻辑，如果发生了大新闻，Agent 权重升高
                w_agent = min(math.pow(abs(agent_mock_score), 2.0), 1.0)
                if abs(agent_mock_score) > 0.7:
                    w_agent = max(w_agent, 0.8)
                    
                w_lstm = 1.0 - w_agent
                final_score = w_lstm * lstm_mock_score + w_agent * agent_mock_score
                
                next_ret = row['next_ret']
                # 过滤掉真实的涨跌停板无效数据（如果是 0 或者极端值）
                if abs(next_ret) < 0.21: 
                    data.append((final_score, next_ret))
                    
        except Exception as e:
            continue
            
    print(f"   [系统] 成功提取了 {len(data)} 条真实历史日线交易样本！")
    return data

def simulate_sharpe_ratio(data, thresholds):
    """
    给定阈值，跑一遍回测，计算策略的简易夏普率或总收益
    thresholds 格式：(strong_buy_th, buy_th, sell_th, strong_sell_th)
    要求: strong_buy > buy > sell > strong_sell
    """
    s_buy, buy, sell, s_sell = thresholds
    if not (s_buy > buy and buy > sell and sell > s_sell):
        return -999.0 # 无效的阈值排序
        
    portfolio_returns = []
    
    for score, true_ret in data:
        # 执行动作决策
        position = 0.0
        if score >= s_buy:
            position = 1.0     # 强力看多，满仓
        elif score >= buy:
            position = 0.5     # 轻仓试盘
        elif score > sell:
            position = 0.0     # 空仓观望 (Hold)
        elif score > s_sell:
            position = -0.5    # 轻仓融券做空
        else:
            position = -1.0    # 强力看空，满仓做空
            
        # 扣除滑点和手续费 (假设万分之二)
        cost = abs(position) * 0.0002
        trade_profit = (position * true_ret) - cost
        portfolio_returns.append(trade_profit)
        
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
    # s_buy: 0.4 到 0.8
    # buy:   0.1 到 0.4
    # sell: -0.4 到 -0.1
    # s_sell: -0.8 到 -0.4
    s_buy_range = [0.4, 0.5, 0.6, 0.7, 0.8]
    buy_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    sell_range = [-0.1, -0.15, -0.2, -0.25, -0.3, -0.4]
    s_sell_range = [-0.4, -0.5, -0.6, -0.7, -0.8]
    
    best_sharpe = -999.0
    best_thresh = None
    
    total_combinations = len(s_buy_range) * len(buy_range) * len(sell_range) * len(s_sell_range)
    print(f"📊 即将验证的参数组合总数: {total_combinations} 次跑批")
    
    count = 0
    for sb in s_buy_range:
        for b in buy_range:
            for s in sell_range:
                for ss in s_sell_range:
                    count += 1
                    thresh = (sb, b, s, ss)
                    sharpe = simulate_sharpe_ratio(data, thresh)
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_thresh = thresh
                        
                    if count % 200 == 0:
                        print(f"   执行进度: {count} / {total_combinations} ...")
                        
    return best_thresh, best_sharpe

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
        best_t, best_s = grid_search_thresholds(real_data)
        
        if best_t is not None:
            best_thresholds_history.append(best_t)
            print(f"   [迭代 {i+1} 最佳结果] Thresholds: {best_t}, Max Sharpe: {best_s:.4f}")
        else:
            print(f"   [迭代 {i+1}] 未找到有效参数。")

    
    print("\n=======================================================")
    print("✅ 【全局参数寻优完成】Optimal Thresholds Found!")
    
    if best_thresholds_history:
        # 整理历史最佳寻找稳定点 (取中位数或平均值)
        # s_buy_range = [0.4, 0.5, 0.6, 0.7, 0.8]
        # buy_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
        # sell_range = [-0.1, -0.15, -0.2, -0.25, -0.3, -0.4]
        # s_sell_range = [-0.4, -0.5, -0.6, -0.7, -0.8]
        
        # 使用中位数对极端随机情况更鲁棒
        final_s_buy = np.median([t[0] for t in best_thresholds_history])
        final_buy = np.median([t[1] for t in best_thresholds_history])
        final_sell = np.median([t[2] for t in best_thresholds_history])
        final_s_sell = np.median([t[3] for t in best_thresholds_history])
        
        import json
        with open("best_t.json", "w", encoding="utf-8") as f:
            json.dump({
                "history": best_thresholds_history,
                "final": [final_s_buy, final_buy, final_sell, final_s_sell]
            }, f, indent=4)
        print("\n   [建议固化进入 dynamic_gating.py 的硬核交易阈值 (基于10次迭代中位数)]")
        print(f"   STRONG BUY  >= {final_s_buy:.2f}    (原设定为 0.60)")
        print(f"   BUY         >= {final_buy:.2f}    (原设定为 0.20)")
        print(f"   SELL         < {final_sell:.2f}    (原设定为 -0.20)")
        print(f"   STRONG SELL  < {final_s_sell:.2f}    (原设定为 -0.60)")
    else:
        print("未收集到足够的阈值数据。")
        
    print("=======================================================")
    print(f"⏱️ 寻优总耗时: {time.time() - start_time:.2f} 秒")
