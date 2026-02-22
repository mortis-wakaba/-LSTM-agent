# -*- coding: utf-8 -*-
"""
mock_interfaces.py

本模块用于模拟队友 A （负责图谱与数据）和队友 C （负责模型与前端）
未来会提供的数据接口。按照“接口先行”的开发红线，保障独立无阻塞开发。
"""

import random
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseGraphProvider(ABC):
    """
    抽象图谱提供者基类。
    强制解耦 Agent 逻辑与底层图谱的具体实现。
    """
    @abstractmethod
    def get_neighbors(self, node_name: str) -> List[Dict]:
        """
        获取节点的直接相邻节点信息。

        参数:
            node_name (str): 目标节点名称
        
        返回:
            List[Dict]: 邻居列表，每个字典必须包含:
                - "name": str, 邻居名称
                - "relation": str, 关系类型 (如 'SUPPLY', 'COMPETE', 'HOLDING')
                - "base_weight": float, 基础影响权重，范围 [-1.0, 1.0]
                  (负数代表负相关/如竞争利空，正数代表正相关/如供应链利好)
        """
        pass

class MockTeammateA(BaseGraphProvider):
    """
    负责图谱与数据（队友 A）的模拟接口。
    """
    def __init__(self):
        # 预先生成 50 只算力股和固定的随机关联，以供稳定测试
        self._stocks = [f"算力股_{i:02d}" for i in range(1, 48)] + ["天孚通信", "中际旭创", "工业富联"]
        self._graph = self._generate_mock_graph()

    def _generate_mock_graph(self) -> dict:
        """
        内部方法：随机生成包含 50 只算力股票及彼此错综复杂关系的网络图谱。
        由于 base_weight 已经包含正负向逻辑，权重将散布在 [-1.0, 1.0]。
        """
        graph = {stock: [] for stock in self._stocks}
        relations = ["SUPPLY", "COMPETE", "HOLDING", "SUBSIDIARY", "PARTNERSHIP"]
        
        # 为每个节点生成 1~4 个邻居
        for node in self._stocks:
            num_neighbors = random.randint(1, 4)
            # 从其余节点中挑选邻居，避免自环
            candidates = [s for s in self._stocks if s != node]
            neighbors = random.sample(candidates, num_neighbors)
            
            for nb in neighbors:
                rel = random.choice(relations)
                # 设定：竞争关系必定为负向连带，控股或子公司通常正规高度正相关，供应链随机偏正相关
                if rel == "COMPETE":
                    weight = random.uniform(-1.0, -0.3)
                elif rel in ["HOLDING", "SUBSIDIARY"]:
                    weight = random.uniform(0.6, 1.0)
                else: 
                    # 供应链/合作
                    weight = random.uniform(0.2, 0.9)
                    
                graph[node].append({
                    "name": nb,
                    "relation": rel,
                    "base_weight": round(weight, 4)
                })
                
        # 强行注入一条确定的测试主干路径，以保证极客风格演示的可追溯性：
        # 天孚通信 (受利好爆发) -> (SUPPLY, 0.9) -> 中际旭创 -> (COMPETE, -0.6) -> 算力股_01
        # 确保原有随机网络不冲突覆盖这部分
        graph["天孚通信"] = [
            {"name": "中际旭创", "relation": "SUPPLY", "base_weight": 0.9},
            {"name": "算力股_10", "relation": "HOLDING", "base_weight": 0.8}
        ]
        graph["中际旭创"] = [
            {"name": "工业富联", "relation": "SUPPLY", "base_weight": 0.5},
            {"name": "算力股_01", "relation": "COMPETE", "base_weight": -0.6}
        ]
        
        return graph

    def get_graph_snapshot(self) -> dict:
        """
        返回包含 50 只算力股完整关联的字典快照。
        """
        return self._graph

    def get_neighbors(self, node_name: str) -> List[Dict]:
        """
        实现抽象基类：抓取节点邻居。
        """
        return self._graph.get(node_name, [])

    def get_latest_news(self, scenario: str = "default") -> dict:
        """
        模拟获取 LLM 抽取的突发新闻。并将强度、情感整合为唯一的 impact_score。
        
        参数:
            scenario (str): 模拟的场景名。
            
        返回:
            dict: 包含 'target_stock', 'impact_score' ([-1.0, 1.0]), 'news_text'
        """
        if scenario == "holiday_shock":
            return {
                "target_stock": "天孚通信",
                "impact_score": 0.95,  # 极性为正，强度达到 0.95 的超级利好
                "news_text": "长假期间重磅利好：OpenAI 正式发布新一代具备强大推理能力的大模型，预计算力硬件需求暴增！"
            }
        
        # 默认随机生成强度合并特征
        return {
            "target_stock": random.choice(self._stocks),
            "impact_score": round(random.uniform(-0.8, 0.8), 2),
            "news_text": "算力行业日常新闻简报..."
        }

class MockTeammateC:
    """
    负责模型与前端（队友 C）的模拟接口。
    """
    def get_lstm_features(self, stock: str) -> List[float]:
        """
        获取 LSTM 模型基于该股票近期量价特征输出的隐状态表示（Hidden State Vector）。
        模拟双向交叉注意力所需的特征向量输入。
        
        参数:
            stock (str): 目标股票名称
            
        返回:
            List[float]: 长度为 64 维的特征向量，模拟模型提取的量价深层特征。
                         此处将伪造一组包含微弱方向倾向的浮点数组合。
        """
        # 为了演示模式A的向下兼容，我们假设 64-d 的第 0 维为主预测趋势分 (-1~1)
        # 后续 63 维为高维模糊表示
        
        # 针对特定股票塞一点手工倾向，保证测试结果符合预期逻辑
        base_trend = 0.0
        if stock == "天孚通信":
            base_trend = 0.15
        elif stock == "中际旭创":
            base_trend = 0.20
        elif stock == "工业富联":
            base_trend = -0.05
        else:
            base_trend = random.uniform(-0.1, 0.1)
            
        # 组装 64-dim list
        hidden_state = [base_trend] + [random.uniform(-0.5, 0.5) for _ in range(63)]
        return hidden_state
