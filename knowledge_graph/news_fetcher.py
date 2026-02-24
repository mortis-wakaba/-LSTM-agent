"""
个股新闻抓取模块

数据源（两个接口合并去重）:
  1. 东方财富 np-listapi  — 个股关联资讯流（行情/资金类偏多）
  2. AKShare stock_news_em — 关键词搜索（事件/深度报道偏多，对LLM更有价值）
  3. 东方财富 np-anotice   — 公司公告
"""

import json
import os
import re
import time
import warnings

import pandas as pd
import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

os.environ["CURL_CA_BUNDLE"] = ""

_session = requests.Session()
_session.trust_env = False
_session.headers.update({"User-Agent": "Mozilla/5.0"})


def _market_prefix(code: str) -> str:
    """上交所=1, 深交所=0"""
    return "1" if code.startswith(("6",)) else "0"


def fetch_news_listapi(code: str, page_size: int = 20) -> list[dict]:
    """东方财富个股资讯流"""
    prefix = _market_prefix(code)
    url = "https://np-listapi.eastmoney.com/comm/wap/getListInfo"
    params = {
        "client": "wap",
        "type": 1,
        "mTypeAndCode": f"{prefix}.{code}",
        "pageSize": page_size,
        "pageNo": 1,
    }
    r = _session.get(url, params=params, verify=False, timeout=15)
    data = r.json()
    items = data.get("data", {}).get("list", [])
    return [
        {
            "stock_code": code,
            "title": it.get("Art_Title", ""),
            "time": it.get("Art_ShowTime", ""),
            "source": it.get("Art_MediaName", ""),
            "url": it.get("Art_Url", ""),
        }
        for it in items
    ]


def fetch_news_akshare(code: str) -> list[dict]:
    """AKShare 关键词搜索"""
    import akshare as ak

    try:
        df = ak.stock_news_em(symbol=code)
    except Exception:
        return []

    col_map = {}
    for c in df.columns:
        if "标题" in c:
            col_map["title"] = c
        elif "内容" in c:
            col_map["content"] = c
        elif "时间" in c:
            col_map["time"] = c
        elif "来源" in c:
            col_map["source"] = c
        elif "链接" in c:
            col_map["url"] = c

    results = []
    for _, row in df.iterrows():
        results.append(
            {
                "stock_code": code,
                "title": row.get(col_map.get("title", ""), ""),
                "time": str(row.get(col_map.get("time", ""), "")),
                "source": row.get(col_map.get("source", ""), ""),
                "url": row.get(col_map.get("url", ""), ""),
                "content": row.get(col_map.get("content", ""), ""),
            }
        )
    return results


def fetch_announcements(code: str, page_size: int = 10) -> list[dict]:
    """东方财富公司公告"""
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    params = {
        "page_size": page_size,
        "page_index": 1,
        "ann_type": "A",
        "stock_list": code,
        "f_node": 0,
        "s_node": 0,
    }
    r = _session.get(url, params=params, verify=False, timeout=15)
    data = r.json()
    items = data.get("data", {}).get("list", [])
    return [
        {
            "stock_code": code,
            "title": it.get("title", ""),
            "time": it.get("notice_date", ""),
            "source": "公告",
            "url": "",
        }
        for it in items
    ]


_DATA_PATTERNS = [
    r"主力资金.*净流[入出]",
    r"融资融券.*余额",
    r"融资净买入",
    r"杠杆资金",
    r"龙虎榜",
    r"北向资金",
    r"大宗交易",
    r"股东户数",
    r"限售股解禁",
    r"每日变动",
    r"平均股价",
    r"连续\d+日",
    r"^\d+只.*股",
    r"^\d+股",
    r"元主力资金.*抢筹",
    r"业绩快报",
]
_DATA_RE = re.compile("|".join(_DATA_PATTERNS))


def _classify_news(title: str) -> str:
    """分类：data=资金/行情数据, event=事件/深度报道"""
    return "data" if _DATA_RE.search(title) else "event"


_CONTENT_PATTERN = re.compile(
    r'<div[^>]*class="txtinfos"[^>]*>(.*?)</div>', re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


def _fetch_article_content(url: str, max_len: int = 200) -> str:
    """从东方财富文章页抓取正文前 max_len 字作为摘要"""
    if not url or "eastmoney.com" not in url:
        return ""
    try:
        r = _session.get(url, verify=False, timeout=10)
        r.encoding = "utf-8"
        m = _CONTENT_PATTERN.search(r.text)
        if m:
            text = _TAG_RE.sub("", m.group(1)).strip()
            text = re.sub(r"\s+", " ", text)
            return text[:max_len]
        ps = re.findall(r"<p[^>]*>(.*?)</p>", r.text, re.DOTALL)
        texts = [_TAG_RE.sub("", p).strip() for p in ps if len(_TAG_RE.sub("", p).strip()) > 20]
        if texts:
            return " ".join(texts[:3])[:max_len]
    except Exception:
        pass
    return ""


def backfill_content(
    news_list: list[dict], delay: float = 0.3, max_len: int = 200
) -> list[dict]:
    """给缺少 content 的新闻补抓正文摘要"""
    for item in news_list:
        if not item.get("content"):
            item["content"] = _fetch_article_content(item.get("url", ""), max_len)
            if delay > 0:
                time.sleep(delay)
    return news_list


def fetch_stock_news_merged(
    code: str, page_size: int = 20, scrape_content: bool = True
) -> list[dict]:
    """合并两个数据源的新闻，过滤垃圾，去重，补摘要"""
    news1 = fetch_news_listapi(code, page_size=page_size)
    news2 = fetch_news_akshare(code)

    seen_titles = set()
    merged = []
    for item in news2 + news1:
        title = item["title"].strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            item["news_type"] = _classify_news(title)
            merged.append(item)

    if scrape_content:
        event_news = [n for n in merged if n.get("news_type") == "event"]
        backfill_content(event_news, delay=0.2, max_len=200)

    merged.sort(key=lambda x: x.get("time", ""), reverse=True)
    return merged


def fetch_all_pool_news(
    codes: list[str], page_size: int = 20, delay: float = 1.0
) -> pd.DataFrame:
    """批量抓取股票池所有个股新闻"""
    all_news = []
    for i, code in enumerate(codes):
        try:
            news = fetch_stock_news_merged(code, page_size=page_size)
            all_news.extend(news)
            print(f"  [{i+1}/{len(codes)}] {code}: {len(news)} 条")
        except Exception as e:
            print(f"  [{i+1}/{len(codes)}] {code}: 失败 - {e}")
        if i < len(codes) - 1:
            time.sleep(delay)

    df = pd.DataFrame(all_news)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time", ascending=False).reset_index(drop=True)
    return df
