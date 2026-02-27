import json
import os
import pandas as pd
from datetime import datetime, timedelta

# Import the existing agent and knowledge graph logic
import sys
# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.reasoning import FinancialAgent
from knowledge_graph.dynamic_graph import DynamicKnowledgeGraph
from knowledge_graph.graph_adapter import RealGraphProvider
from knowledge_graph.stock_pool import get_all_stocks_flat

def main():
    base_dir = r"c:\Users\mortis\Desktop\ai4f\-LSTM-agent"
    data_dir = os.path.join(base_dir, "data")
    relations_path = os.path.join(data_dir, "aligned_relations.json")
    output_path = os.path.join(data_dir, "historical_agent_scores.csv")

    if not os.path.exists(relations_path):
        print(f"File not found: {relations_path}")
        return

    with open(relations_path, "r", encoding="utf-8") as f:
        relations = json.load(f)

    # Convert relations into a dictionary grouped by date
    relations_by_date = {}
    for r in relations:
        date = r.get("date")
        if not date:
            continue
        if date not in relations_by_date:
            relations_by_date[date] = []
        relations_by_date[date].append(r)

    # Get sorted dates
    all_dates = sorted(list(relations_by_date.keys()))
    if not all_dates:
        print("No valid dates found in relations.")
        return

    start_date = all_dates[0]
    end_date = all_dates[-1]
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Initialize Graph and Agent
    print(f"Initializing Graph and Agent...")
    static_graph_provider = RealGraphProvider() # This automatically builds the initial DKG
    agent = FinancialAgent()

    all_stock_codes = [s[0] for s in get_all_stocks_flat()]
    
    # Store results: List of dicts {date, stock_code, agent_score(total), direct_score, indirect_score}
    historical_scores = []

    print(f"Running simulation from {start_date} to {end_date}...")
    
    # Iterate through every single day (including weekends/holidays to decay properly)
    curr_dt = start_dt
    while curr_dt <= end_dt:
        date_str = curr_dt.strftime("%Y-%m-%d")
        
        # 1. Apply news events for today
        todays_events = relations_by_date.get(date_str, [])
        for event in todays_events:
            source = event["source"]
            target = event["target"]
            relation = event["relation"]
            sentiment = event.get("sentiment", 0.0)
            
            # Update the dynamic graph with the new relation/sentiment
            added = static_graph_provider.dkg.add_edge(source, target, relation, sentiment=sentiment)
            if not added:
                static_graph_provider.dkg.update_edge_sentiment(source, target, sentiment)
            
            # Trigger the propagation
            agent.propagate_impact(target_stock=source, initial_power=sentiment, graph_provider=static_graph_provider)
            # If the news mentions a target, also propagate from target directly (optional, but realistic)
            if source != target and target in all_stock_codes:
                 agent.propagate_impact(target_stock=target, initial_power=sentiment, graph_provider=static_graph_provider)

        # 2. Record scores for all 50 stocks for this date
        for stock_code in all_stock_codes:
            scores = agent.get_feature_vectors(stock_code)
            # scores is a list: [total_score]
            total_score = scores[0] if scores else 0.0
            historical_scores.append({
                "date": date_str,
                "stock_code": stock_code,
                "total_score": round(total_score, 4)
            })

        # 3. Apply daily AR(1) decay at the end of the day
        agent.daily_decay()
        
        curr_dt += timedelta(days=1)

    # Save to CSV
    df = pd.DataFrame(historical_scores)
    df.to_csv(output_path, index=False)
    print(f"Finished. Generated {len(df)} score records.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
