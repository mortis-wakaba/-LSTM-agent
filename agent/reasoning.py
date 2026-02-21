# -*- coding: utf-8 -*-
"""
reasoning.py

核心推理引擎：负责基于知识图谱与突发事件进行多跳衰减传导计算。
"""

class GraphAgent:
    """
    基于图谱的推理 Agent，能够根据新闻冲击计算直接和间接信号。
    """
    def __init__(self, decay_lambda: float = 0.8):
        """
        初始化 Agent。
        
        参数:
            decay_lambda (float): 多跳传递的衰减系数（λ），默认 0.8。
        """
        self.decay_lambda = decay_lambda

    def calculate_signals(self, news_triple: dict, graph_snapshot: dict) -> dict:
        """
        计算新闻对个股及产业链关联个股的综合影响信号分数。
        
        计算公式：
        1. 直接冲击：Direct_Signal = impact_score * sentiment
        2. 单跳传递计算：对于由图谱连接的节点，间接冲击 Indirect_Signal = Direct_Signal * base_weight * λ
        
        参数:
            news_triple (dict): 包含 'target_stock', 'impact_score', 'sentiment' 的字典。
            graph_snapshot (dict): 当前图谱结构的拓扑映射。
            
        返回:
            dict: key 为股票名称，value 为计算得出的 agent_signal (范围 [-1.0, 1.0])。
        """
        target = news_triple.get("target_stock")
        impact_score = news_triple.get("impact_score", 0.0)
        sentiment = news_triple.get("sentiment", 0)
        
        # -------------------------------------------------------------
        # 1. 计算直接冲击信号 (Direct_Signal)
        # -------------------------------------------------------------
        # 公式：Direct_Signal = impact_score * sentiment
        direct_signal = impact_score * sentiment
        
        # 记录各节点分数
        signals = {target: direct_signal}
        
        if target not in graph_snapshot:
            return signals
            
        # -------------------------------------------------------------
        # 2. 计算单跳衰减推理 (Indirect_Signal)
        # -------------------------------------------------------------
        # 查找目标股票的上游和下游节点，计算波及影响
        relations = graph_snapshot[target]
        all_connected = relations.get("upstream", []) + relations.get("downstream", [])
        
        for edge in all_connected:
            linked_node = edge["node"]
            base_weight = edge["base_weight"]
            
            # 公式：Indirect_Signal = Direct_Signal * base_weight * λ
            indirect_signal = direct_signal * base_weight * self.decay_lambda
            
            # 控制上下限在 [-1.0, 1.0] 范围内，防止极值溢出
            indirect_signal = max(min(indirect_signal, 1.0), -1.0)
            
            # 记录波及个股的分数
            # （注：如果同一个节点既是上游又是下游，这里作简单的情况覆盖，实际应用中可酌情累加或求最大绝对值）
            signals[linked_node] = indirect_signal
            
        return signals
