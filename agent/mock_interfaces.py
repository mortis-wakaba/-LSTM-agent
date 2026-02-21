# -*- coding: utf-8 -*-
"""
mock_interfaces.py

本模块用于模拟队友 A （负责图谱与数据）和队友 C （负责模型与前端）
未来会提供的数据接口。按照“接口先行”的开发红线，保障独立无阻塞开发。
"""

import random

class MockTeammateA:
    """
    负责图谱与数据（队友 A）的模拟接口。
    """
    def get_graph_snapshot(self) -> dict:
        """
        获取当前算力股票的基础物理关联图谱拓扑结构。
        
        返回:
            dict: 节点的关联关系字典，包含上下游(upstream/downstream)关系及基础关系权重(base_weight)。
        """
        return {
            "天孚通信": {
                "upstream": [{"node": "光芯片供应商X", "base_weight": 0.6}],
                "downstream": [{"node": "中际旭创", "base_weight": 0.9}]
            },
            "中际旭创": {
                "upstream": [{"node": "天孚通信", "base_weight": 0.9}],
                "downstream": [{"node": "北美云厂商A", "base_weight": 0.95}, {"node": "工业富联", "base_weight": 0.5}]
            },
            "工业富联": {
                "upstream": [{"node": "中际旭创", "base_weight": 0.5}, {"node": "GPU核心供应商N", "base_weight": 0.95}],
                "downstream": [{"node": "国内服务器厂商Y", "base_weight": 0.7}]
            }
        }

    def get_latest_news_triple(self, scenario: str = "default") -> dict:
        """
        模拟获取 LLM 抽取的突发新闻三元组数据。
        
        参数:
            scenario (str): 模拟的场景名，用于快速测试不同分支。
            
        返回:
            dict: 包含受冲击股票(target_stock)、冲击力得分(impact_score)、
                  情感极性(sentiment)及新闻原文(news_text)。
        """
        if scenario == "holiday_shock":
            return {
                "target_stock": "天孚通信",
                "impact_score": 0.95,  # 极高冲击力
                "sentiment": 1,        # 极性：1为利好，-1为利空
                "news_text": "长假期间重磅利好：OpenAI 正式发布新一代具备强大推理能力的大模型，预计算力硬件需求暴增！"
            }
        
        # 默认随机生成一条新闻
        return {
            "target_stock": random.choice(["中际旭创", "工业富联"]),
            "impact_score": random.uniform(0.3, 0.7),
            "sentiment": random.choice([-1, 1]),
            "news_text": "算力行业日常新闻简报..."
        }

class MockTeammateC:
    """
    负责模型与前端（队友 C）的模拟接口。
    """
    def get_lstm_prediction(self, stock: str) -> float:
        """
        获取 LSTM 模型基于历史量价数据的趋势预测分数。
        
        参数:
            stock (str): 目标股票名称
            
        返回:
            float: 预测分数，范围 [-1.0, 1.0]。
        """
        # 针对休市长假等场景，假设 LSTM 因为缺乏最新交易日数据，
        # 给出的是相对保守或滞后的预测（维持停盘前的微弱趋势）
        mock_predictions = {
            "天孚通信": 0.15,
            "中际旭创": 0.20,
            "工业富联": -0.05,
        }
        return mock_predictions.get(stock, 0.0)
