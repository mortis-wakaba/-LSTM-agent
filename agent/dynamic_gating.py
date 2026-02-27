# -*- coding: utf-8 -*-
"""
dynamic_gating.py

动态门控融合模块：负责融合 LSTM 历史模型分数和 Agent 即时多跳分数的最终中枢决策。
包含两种可配置的融合模式：数学门控驱动 (Mode A) 与 双向注意力适配机制 (Mode B)。
"""

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionGate(nn.Module):
    """
    基于 PyTorch 的交叉注意力门控机制 (Cross-Attention Gating)。
    接收 LSTM 的 64 维隐藏状态特征作为 Query/Key，Agent 特征作为 Value 构建动态融合权重。
    """
    def __init__(self, lstm_hidden_dim=64, agent_dim=1):
        super().__init__()
        # 用于对齐维度的线性投影层
        self.agent_proj = nn.Linear(agent_dim, lstm_hidden_dim)
        
        # 将融合特征转换为注意力的线性层
        self.attention_net = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # 输出 2 维：[w_lstm_raw, w_agent_raw]
        )
        
    def forward(self, lstm_h: torch.Tensor, agent_score: torch.Tensor):
        """
        lstm_h: [batch, 64]
        agent_score: [batch, 1]
        返回: 归一化后的注意力权重 [batch, 2] -> (w_lstm, w_agent)
        """
        # 将 agent 1维特征投影到与 LSTM 相同的 64 维子空间
        agent_h = self.agent_proj(agent_score)
        
        # 计算特征交叉
        # 在这里，我们将 LSTM 状态和投影后的 Agent 状态进行拼接
        combined = torch.cat([lstm_h, agent_h], dim=-1)
        
        # 生成注意力并归一化
        attn_logits = self.attention_net(combined)
        weights = F.softmax(attn_logits, dim=-1)
        
        return weights

class FusionEngine:
    """
    负责动态权重分配与最终买卖量化信号映射的引挚。
    """
    def __init__(self, mode: str = 'math', k: float = 2.0, load_attention_weights: str = None):
        """
        初始化动态门控引擎。
        
        参数:
            mode (str): 'math' 为数学公式门控，'attention' 为双轨注意力适配。
            k (float): 数学模式下的超参数，用于控制 Agent 分数的非线性放大力度。
            load_attention_weights (str): 预训练的 CrossAttentionGate 权重路径
        """
        if mode not in ['math', 'attention']:
            raise ValueError("mode 必须是 'math' 或 'attention'")
        self.mode = mode
        self.k = k
        
        self.attention_gate = CrossAttentionGate(lstm_hidden_dim=64, agent_dim=1)
        if load_attention_weights and mode == 'attention':
            self.attention_gate.load_state_dict(torch.load(load_attention_weights))
        self.attention_gate.eval() # 默认推理模式

    def calculate_final_score(self, lstm_features: List[float], agent_features: List[float]) -> dict:
        """
        根据指定的融合机制，计算个股最终得分。
        
        参数:
            lstm_features (List[float]): LSTM 模型的 64 维隐状态输出向量。
                                         我们约定第 0 维代表趋势主预测分 [-1.0, 1.0]。
            agent_features (List[float]): Agent 给出的 1 维特征向量 [total_score]。
            
        返回:
            dict: 包含 final_score (最终得分) 与 action (操作指令)、状态特征等。
        """
        lstm_score_scalar = lstm_features[0]
        agent_score_scalar = agent_features[0]
        
        if self.mode == 'math':
            result = self._fusion_math_mode(lstm_score_scalar, agent_score_scalar)
        else:
            result = self._fusion_attention_mode(lstm_features, agent_features)
            
        # 将连续的 -1 到 +1 分数映射到实际的离散交易动作
        final_score = result['final_score']
        action = self._map_score_to_action(final_score)
        result['action'] = action
        result['mode'] = self.mode
        
        return result
        
    def _fusion_math_mode(self, lstm_score: float, agent_score: float) -> dict:
        """
        模式 A (数学门控)：
        W_event = |S_agent|^k
        Final_score = (1 - W_event) * LSTM_score + W_event * Agent_score
        """
        # 基础动态赋权公式
        w_agent = math.pow(abs(agent_score), self.k)
        # 确保权重不过界
        w_agent = min(w_agent, 1.0)
        
        # 利用自身连续打分（而非离散标志位）决定接管阈值
        if abs(agent_score) > 0.7:
            w_agent = max(w_agent, 0.8)
            status = "重大事件/断档接管 (Math)"
        else:
            status = "常态化基础融合 (Math)"
            
        w_lstm = 1.0 - w_agent
        
        final_score = w_lstm * lstm_score + w_agent * agent_score
        
        return {
            "final_score": round(final_score, 4),
            "status": status,
            "weights": {"w_lstm": w_lstm, "w_agent": w_agent}
        }
        
    def _fusion_attention_mode(self, lstm_features: List[float], agent_features: List[float]) -> dict:
        """
        模式 B (注意力适配)：
        如果可用，使用 PyTorch 的 CrossAttentionGate 推理出融合权重。
        """
        lstm_score = lstm_features[0]
        agent_score = agent_features[0]
        
        # 兼容老的回测如果只有 1 维 (无隐藏状态)，回退到原始 mock
        if len(lstm_features) == 1:
            return self._fusion_math_mode(lstm_score, agent_score)
            
        # 提取 64 维隐藏状态
        hidden_states = lstm_features[1:]
        
        # 转换为张量进行推理
        with torch.no_grad():
            lstm_h_tensor = torch.tensor([hidden_states], dtype=torch.float32)
            agent_score_tensor = torch.tensor([[agent_score]], dtype=torch.float32)
            
            weights = self.attention_gate(lstm_h_tensor, agent_score_tensor)[0].numpy()
            
        w_lstm = float(weights[0])
        w_agent = float(weights[1])
        
        final_score = w_lstm * lstm_score + w_agent * agent_score
        status = "交叉注意力对齐 (Neural)"
        
        return {
            "final_score": round(final_score, 4),
            "status": status,
            "weights": {"w_lstm": w_lstm, "w_agent": w_agent}
        }

    def _map_score_to_action(self, score: float) -> str:
        """
        将连续的分数映射为具体的离散交易信号。
        (阈值由历史网格搜索回测寻优产生)
        """
        if score >= 0.40:
            return "STRONG BUY"
        elif score >= 0.15:
            return "BUY"
        elif score > -0.40:
            return "HOLD"
        elif score > -0.70:
            return "SELL"
        else:
            return "STRONG SELL"
