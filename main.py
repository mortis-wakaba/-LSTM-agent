# -*- coding: utf-8 -*-
"""
main.py

项目核心测试入口：串联图谱数据、LSTM预测流、图谱推理传导及动态门控融合引擎，进行最终链路联测。
展示了系统在“长假数据断档接管/突发事件”极端场景下的动态决策流转全过程。
支持可配置的融合模式体验。
"""

import time
import argparse
import sys
import io

# 强制输出设为 utf-8，防止 Windows GBK 终端下特殊表情符号报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent.mock_interfaces import MockTeammateA, MockTeammateC
from agent.reasoning import FinancialAgent
from agent.dynamic_gating import FusionEngine

def print_banner():
    banner = """
    =============================================================================
     🌟 [内生-外生双轨融合架构] AI 算力资产定价系统 (Agentic Pricing Engine) 🌟
    =============================================================================
    """
    print(banner)

def simulate_pipeline(fusion_mode='attention'):
    print_banner()
    
    # ---------------- 1. 初始化依赖模块 ----------------
    print(f"\n[Sys] 🤖 引导程序启动中... (预选融合网络模式: {fusion_mode.upper()})")
    time.sleep(0.5)
    
    mock_a = MockTeammateA()
    mock_c = MockTeammateC()
    financial_agent = FinancialAgent(decay_lambda=0.8)
    fusion_engine = FusionEngine(mode=fusion_mode, k=2.0)
    
    print("[Sys] ✅ Agent 中枢推理引擎加载完毕，依赖接口（基于 BaseGraphProvider 注入）检查正常。")
    time.sleep(0.5)
    
    # ---------------- 2. 拉取前置图谱与数据 ----------------
    print("\n[Node A] 正在从知识图谱生成/抽取 A 股算力产业链物理关联拓扑...")
    time.sleep(0.5)
    graph_snapshot = mock_a.get_graph_snapshot()
    # 仅展示开头几个核心支点
    keys_sample = list(graph_snapshot.keys())[-3:] + ["..."]
    print(f"[Node A] 🌐 成功获取包含 50 个节点的图谱，重点覆盖：{keys_sample}")
    
    print("\n[Node A] 正在监控全网实时事件与突发新闻...")
    time.sleep(1)
    # 抽取特殊设定：模拟休市期间的重磅利好新闻
    news_dict = mock_a.get_latest_news(scenario="holiday_shock")
    
    print(f"\n==================== 🚨 捕获核心突发新闻 🚨 ====================")
    print(f"📄 新闻原文: {news_dict['news_text']}")
    print(f"🎯 直接冲击目标: {news_dict['target_stock']}")
    print(f"💥 基础冲击力度 (Impact Score): {news_dict['impact_score']}")
    print("================================================================")
    
    # ---------------- 3. 基于图谱的多跳推演 ----------------
    print("\n[Agent] 🧠 触发《多智能体动态图谱推理引擎》进行穿透推演...")
    time.sleep(0.8)
    
    # 触发一次每日衰减
    print("[Agent] ⏳ 触发一次 AR(1) 记忆衰减判定 (daily_decay)...")
    financial_agent.daily_decay()
    time.sleep(0.5)

    # 执行广度优先搜索传导冲击
    financial_agent.propagate_impact(
        target_stock=news_dict['target_stock'],
        initial_power=news_dict['impact_score'],
        graph_provider=mock_a
    )
    
    print("[Agent] 📉 最新产业链节点冲击分数计算完毕，追踪受影响链路：")
    for stock, score in financial_agent.scores.items():
        if abs(score) >= 0.05:
            print(f"       -> 实体 [{stock}]: {score:+.4f}")
            
    # ---------------- 4. 动态门控融合与决策输出 ----------------
    print("\n[Gateway] ⚖️ 进入动态门控融合网络 (Dynamic Gating Fusion)...")
    time.sleep(0.5)
    print("[Gateway] 📡 正在拉取底层 LSTM 特征并与 Agent 特征向量对齐计算...")
    
    print("\n========== 最终系统生成交易指令 (Final Trade Actions) ==========")
    
    demo_stocks = ["天孚通信", "中际旭创", "算力股_01", "工业富联"]
    
    for stock in demo_stocks:
        # 1. 获取模型特征 (64-d) 和 Agent 提纯特征 (2-d)
        lstm_feat = mock_c.get_lstm_features(stock)
        agent_feat = financial_agent.get_feature_vectors(stock)
        
        # 2. 调用引擎融合计算
        result = fusion_engine.calculate_final_score(
            lstm_features=lstm_feat, 
            agent_features=agent_feat
        )
        
        # 3. 极客风展示单只股票的计算穿透信息
        lstm_display = lstm_feat[0]
        agent_display = agent_feat[0]
        print(f"🔹 标的: 【{stock}】| LSTM 特征层_0: {lstm_display:+.2f} | Agent 记忆强度: {agent_display:+.4f}")
        print(f"   [!] 引擎激活状态: {result['status']}")
        print(f"   [#] 动态门控权重 >> LSTM侧: {result['weights']['w_lstm']*100:.1f}% | Agent侧: {result['weights']['w_agent']*100:.1f}%")
        print(f"   [=>] 最终定价层对齐得分: {result['final_score']}")
        print(f"   [=>] 📊 离散化交易信号动作: ===> {result['action']} <===")
        print("-" * 75)

    print(f"\n[Sys] 🏁 {fusion_mode.upper()} 模式核心 Agent 中枢链路全量联测结束。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent Pricing System Demo")
    parser.add_argument("--mode", type=str, default="attention", choices=["math", "attention"], 
                        help="融合引擎类型选择 (math 或 attention)")
    args = parser.parse_args()
    
    simulate_pipeline(fusion_mode=args.mode)
