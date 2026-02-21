# -*- coding: utf-8 -*-
"""
dynamic_gating.py

动态门控融合模块：负责融合 LSTM 历史模型分数和 Agent 即时多跳分数的最终中枢决策。
实现了“内生-外生双轨融合架构”的核心权重量化分配逻辑。
"""

class FusionEngine:
    """
    负责动态权重分配与最终买卖量化信号映射的引挚。
    """
    
    def calculate_final_score(self, lstm_score: float, agent_signal: float, is_holiday_aftermath: bool = False) -> dict:
        """
        根据动态门控机制计算个股最终得分。
        
        逻辑要求（双轨融合门控算法）：
        - 基础状态： 最终得分 = 0.7 * lstm_score + 0.3 * agent_signal
        - 特例1 (长假断档接管)：当 is_holiday_aftermath=True 时，权重切换为 0.1 * LSTM + 0.9 * Agent
        - 特例2 (核弹级事件夺权)：当 abs(agent_signal) > 0.8 时，Agent 强制夺权： 0.3 * LSTM + 0.7 * Agent
        （注：如果特例1和特例2同时满足，代码优先让长假断档特例起判断主导，因其宏观背景更为确凿。）
        
        参数:
            lstm_score (float): 队友 C 的 LSTM 趋势预测分数 [-1.0, 1.0]
            agent_signal (float): 队友 B 的图谱推理得出的短期突发影响分数 [-1.0, 1.0]
            is_holiday_aftermath (bool): 是否处于长假休市刚结束，数据断档的特殊时期
            
        返回:
            dict: 包含 final_score (最终得分) 与 action (操作指令) 的字典。
        """
        
        # -------------------------------------------------------------
        # 确定动态权重 (w_lstm, w_agent)
        # -------------------------------------------------------------
        if is_holiday_aftermath:
            # 特例 1：长假数据断档接管
            # 此时量价数据停留在节前，缺乏时效性，Agent 短期舆情逻辑占据绝对主导
            w_lstm, w_agent = 0.1, 0.9
            status = "长假数据断档接管"
        elif abs(agent_signal) > 0.8:
            # 特例 2：核弹级事件夺权
            # 当发生行业颠覆级制裁、里程碑式技术突破等极端事件时，Agent 临时夺回定价权主导
            w_lstm, w_agent = 0.3, 0.7
            status = "核弹级事件临时夺权"
        else:
            # 基础状态：常态化融合
            # 市场平稳期，由基于量价连续演变的 LSTM 占据主要计算权重
            w_lstm, w_agent = 0.7, 0.3
            status = "常态化基础融合"
            
        # -------------------------------------------------------------
        # 计算最终得分 (Final Score)
        # -------------------------------------------------------------
        final_score = w_lstm * lstm_score + w_agent * agent_signal
        
        # 将连续的 -1 到 +1 分数映射到实际的离散交易动作
        action = self._map_score_to_action(final_score)
        
        return {
            "final_score": round(final_score, 4),
            "action": action,
            "status": status,
            "weights": {"w_lstm": w_lstm, "w_agent": w_agent}
        }
        
    def _map_score_to_action(self, score: float) -> str:
        """
        将连续的分数映射为具体的离散交易信号。
        
        规则参考（阈值划分）：
            score >= 0.6            -> STRONG BUY (强烈买入)
            0.2 <= score < 0.6      -> BUY (买入)
            -0.2 < score < 0.2      -> HOLD (持有/观望)
            -0.6 < score <= -0.2    -> SELL (卖出)
            score <= -0.6           -> STRONG SELL (强烈卖出)
        """
        if score >= 0.6:
            return "STRONG BUY"
        elif score >= 0.2:
            return "BUY"
        elif score > -0.2:
            return "HOLD"
        elif score > -0.6:
            return "SELL"
        else:
            return "STRONG SELL"
