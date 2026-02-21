# -*- coding: utf-8 -*-
"""
main.py

项目核心测试入口：串联图谱数据、LSTM预测流、图谱推理传导及动态门控融合引擎，进行最终链路联测。
展示了系统在“长假数据断档接管”极端场景下的动态决策流转全过程。
"""

import time
from agent.mock_interfaces import MockTeammateA, MockTeammateC
from agent.reasoning import GraphAgent
from agent.dynamic_gating import FusionEngine

def print_banner():
    banner = """
    =============================================================================
     🌟 [内生-外生双轨融合架构] AI 算力资产定价系统 (Agentic Pricing Engine) 🌟
    =============================================================================
    """
    print(banner)

def simulate_pipeline():
    print_banner()
    
    # ---------------- 1. 初始化依赖模块 ----------------
    print("\n[Sys] 🤖 引导程序启动中...")
    time.sleep(0.5)
    
    mock_a = MockTeammateA()
    mock_c = MockTeammateC()
    graph_agent = GraphAgent(decay_lambda=0.8)
    fusion_engine = FusionEngine()
    
    print("[Sys] ✅ Agent 中枢推理引擎加载完毕，各类依赖接口（Mock）检查正常。")
    time.sleep(0.5)
    
    # ---------------- 2. 拉取前置图谱与数据 ----------------
    print("\n[Node A] 正在从知识图谱抽取 A 股算力产业链物理关联拓扑...")
    time.sleep(0.5)
    graph_snapshot = mock_a.get_graph_snapshot()
    print(f"[Node A] 🌐 成功获取图谱，包含核心支点：{list(graph_snapshot.keys())}")
    
    print("\n[Node A] 正在监控全网实时事件与突发新闻...")
    time.sleep(1)
    # 抽取特殊设定：模拟休市期间的重磅利好新闻
    news_triple = mock_a.get_latest_news_triple(scenario="holiday_shock")
    
    print(f"\n==================== 🚨 捕获核心突发新闻 🚨 ====================")
    print(f"📄 新闻原文: {news_triple['news_text']}")
    print(f"🎯 直接冲击目标: {news_triple['target_stock']}")
    print(f"💥 冲击强度: {news_triple['impact_score']}  |  ☯️ 情感极性: {news_triple['sentiment']}")
    print("================================================================")
    
    # ---------------- 3. 基于图谱的多跳推演 ----------------
    print("\n[Agent] 🧠 触发《多智能体动态图谱推理引擎》进行穿透推演...")
    time.sleep(0.8)
    # λ衰减系数已默认 0.8，计算出该突发新闻对上下游的连带波及效应
    agent_signals = graph_agent.calculate_signals(news_triple, graph_snapshot)
    
    print("[Agent] 📉 计算出产业链多跳节点冲击信号（Agent Signal）：")
    for stock, sig in agent_signals.items():
        print(f"       -> 实体 [{stock}]: {sig:.4f}")
        
    # ---------------- 4. 动态门控融合与决策输出 ----------------
    print("\n[Gateway] ⚖️ 进入动态门控融合中枢决策模块...")
    time.sleep(0.5)
    print("[Gateway] 📡 正在调用 LSTM 模型接口拉取基础量价连续性分数...")
    
    # 【核心场景设置】：设定当前情境为大假刚结束，启动接管特例逻辑
    is_holiday = True
    print(f"[Gateway] 🔧 注入宏观时序状态参数: is_holiday_aftermath = {is_holiday}")
    
    print("\n========== 最终系统生成交易指令 (Final Trade Actions) ==========")
    for stock, a_sig in agent_signals.items():
        # 1. 拉取队友 C 的 LSTM 历史数据保守预估分
        l_score = mock_c.get_lstm_prediction(stock)
        
        # 2. 调用引擎融合计算
        result = fusion_engine.calculate_final_score(
            lstm_score=l_score, 
            agent_signal=a_sig, 
            is_holiday_aftermath=is_holiday
        )
        
        # 3. 极客风展示单只股票的计算穿透信息
        print(f"🔹 标的: 【{stock}】| LSTM Score: {l_score:+.2f} | Agent Signal: {a_sig:+.4f}")
        print(f"   [!] 引擎门控状态: {result['status']}")
        print(f"   [#] 权重分配比重 - LSTM: {result['weights']['w_lstm']*100:.0f}% | Agent: {result['weights']['w_agent']*100:.0f}%")
        print(f"   [=>] 最终定价综合得分: {result['final_score']}")
        print(f"   [=>] 📊 量化交易动作指令: ===> {result['action']} <===")
        print("-" * 65)

    print("\n[Sys] 🏁 核心 Agent 中枢链路全量测试结束。输出报告将推送到下游券商 API 执行器。")

if __name__ == "__main__":
    simulate_pipeline()
