"""
LLM 新闻分析模块 — 用 Claude API 从新闻中提取公司关系和 sentiment
"""

import json
import re
from pathlib import Path

import anthropic

from knowledge_graph.stock_pool import get_all_stocks_flat

# =====================================================================
#  从 .env 文件读取配置
# =====================================================================

_ENV_FILE = Path(__file__).resolve().parent / ".env"


def _load_env() -> dict[str, str]:
    """读取 knowledge_graph/.env 文件中的配置"""
    config = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config


def get_api_key() -> str:
    """获取 API key"""
    import os
    config = _load_env()
    key = (config.get("ANTHROPIC_API_KEY")
           or os.environ.get("ANTHROPIC_API_KEY")
           or "")
    if not key or not key.isascii() or not key.startswith("sk-"):
        return ""
    return key


def get_base_url() -> str | None:
    """获取 API base_url"""
    config = _load_env()
    return config.get("ANTHROPIC_BASE_URL") or None


# =====================================================================
#  股票池信息（嵌入 prompt）
# =====================================================================

def _build_stock_context() -> str:
    lines = []
    for code, name, sector in get_all_stocks_flat():
        lines.append(f"{code}={name}")
    return "、".join(lines)


STOCK_CONTEXT = _build_stock_context()

# =====================================================================
#  Prompt 模板
# =====================================================================

SYSTEM_PROMPT = f"""从新闻中提取股票池内公司之间的关系，输出JSON数组。禁止输出任何解释文字。

股票池（50只，其他公司忽略）：{STOCK_CONTEXT}

关系类型（5选1）：supply=供应链(有向,source→target)、compete=竞争(无向)、peer=板块联动(无向)、invest=控股(有向)、cooperate=合作(无向)

sentiment∈[-1,1]：正=利好，负=利空，绝对值=强度

输出格式（严格遵守）：
[{{"source":"代码","target":"代码","relation":"类型","sentiment":数值,"description":"一句话"}}]
无关系则输出[]"""

USER_PROMPT_TEMPLATE = """{news_text}

提取上述新闻中的公司关系，只输出JSON："""


# =====================================================================
#  分析器
# =====================================================================

class NewsAnalyzer:
    """用 Claude API 分析新闻，提取公司关系和 sentiment"""

    def __init__(self, api_key: str, base_url: str | None = None,
                 model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url or "https://api.anthropic.com",
            auth_token=None,
        )
        self.model = model

    def analyze(self, news_text: str) -> list[dict]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(news_text=news_text)},
            ],
        )
        raw = message.content[0].text.strip()
        return self._parse_response(raw)

    def analyze_batch(self, news_list: list[str]) -> list[list[dict]]:
        return [self.analyze(news) for news in news_list]

    def _parse_response(self, raw: str) -> list[dict]:
        """解析 LLM 返回的 JSON，容忍混入的解释文字"""
        # 策略1：直接解析
        try:
            results = json.loads(raw.strip())
            if isinstance(results, list):
                return self._validate(results)
        except json.JSONDecodeError:
            pass

        # 策略2：提取代码块
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if m:
            try:
                results = json.loads(m.group(1).strip())
                if isinstance(results, list):
                    return self._validate(results)
            except json.JSONDecodeError:
                pass

        # 策略3：找 [ 到 ] 之间
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                results = json.loads(raw[start:end + 1])
                if isinstance(results, list):
                    return self._validate(results)
            except json.JSONDecodeError:
                pass

        print(f"警告: JSON 解析失败，原始输出:\n{raw[:200]}...")
        return []

    def _validate(self, results: list) -> list[dict]:
        valid = []
        required_keys = {"source", "target", "relation", "sentiment", "description"}
        valid_relations = {"supply", "compete", "peer", "invest", "cooperate"}

        for item in results:
            if not isinstance(item, dict):
                continue
            if not required_keys.issubset(item.keys()):
                continue
            if item["relation"] not in valid_relations:
                continue
            item["sentiment"] = max(-1.0, min(1.0, float(item["sentiment"])))
            valid.append(item)

        return valid


def analyze_and_update(analyzer: NewsAnalyzer, dkg, news_text: str) -> list[dict]:
    """分析一条新闻并更新动态图谱。"""
    results = analyzer.analyze(news_text)

    for r in results:
        added = dkg.add_edge(
            source_code=r["source"],
            target_code=r["target"],
            relation_type=r["relation"],
            description=r["description"],
            sentiment=r["sentiment"],
        )
        if not added:
            dkg.update_edge_sentiment(r["source"], r["target"], r["sentiment"])

    return results
