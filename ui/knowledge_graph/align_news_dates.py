import json
import os
import re

def main():
    base_dir = r"c:\Users\mortis\Desktop\ai4f\-LSTM-agent\data"
    cache_path = os.path.join(base_dir, "all_news_cache.json")
    relations_path = os.path.join(base_dir, "all_extracted_relations.json")
    output_path = os.path.join(base_dir, "aligned_relations.json")

    print("Loading news cache...")
    with open(cache_path, "r", encoding="utf-8") as f:
        news_cache = json.load(f)

    # 建立 "新闻标题" 到 "日期" (YYYY-MM-DD) 的映射
    # 处理标题中可能存在的空格或特殊字符，进行归一化
    def normalize_title(t):
        return re.sub(r'\s+', '', t).lower()

    title_to_date = {}
    for news in news_cache:
        if "title" in news and "time" in news:
            t_norm = normalize_title(news["title"])
            # time 格式: "2026-02-26 13:47:59", 取前 10 位即 YYYY-MM-DD
            date_str = news["time"][:10]
            title_to_date[t_norm] = date_str

    print(f"Loaded {len(title_to_date)} unique normalized news titles with dates.")

    print("Loading relations...")
    with open(relations_path, "r", encoding="utf-8") as f:
        relations = json.load(f)

    aligned = []
    missing_dates = 0

    for r in relations:
        title = r.get("news_title", "")
        t_norm = normalize_title(title)
        
        # 查找日期
        date_str = title_to_date.get(t_norm)
        
        # 如果归一化匹配不上，尝试子串匹配
        if not date_str:
            for k, v in title_to_date.items():
                if len(k) > 10 and (k in t_norm or t_norm in k):
                    date_str = v
                    break
                    
        if date_str:
            r["date"] = date_str
            aligned.append(r)
        else:
            missing_dates += 1

    print(f"Total relations: {len(relations)}")
    print(f"Successfully aligned with dates: {len(aligned)}")
    print(f"Missing dates: {missing_dates}")

    # 按日期排序
    aligned.sort(key=lambda x: x["date"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)
    
    print(f"Saved aligned relations to {output_path}")

if __name__ == "__main__":
    main()
