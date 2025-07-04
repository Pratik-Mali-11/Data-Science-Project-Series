import pandas as pd
import numpy as np

class BankInterventionOptimizer:
    def __init__(self, customer_data):
        self.data = customer_data.copy()
        self._preprocess_data()
        
        # Validated strategy parameters (cost, risk_reduction, max_usage)
        self.strategies = {
            "Wealth Manager Call": (75, 0.50, 3),
            "Credit Limit Increase": (40, 0.35, 5), 
            "Personalized Offer": (25, 0.30, 8),
            "ATM Fee Waiver": (10, 0.15, 10)
        }
    
    def _preprocess_data(self):
        
        self.data['churn_prob'] = np.minimum(self.data['churn_prob'], 0.95)  # Cap at 95%
        self.data['CLV'] = self._calculate_clv()
    
    def _calculate_clv(self):
        
        monthly_profit = self.data['Balance'] * 0.0045  # 0.45% monthly yield
        retention_rate = 1 - self.data['churn_prob']
        return monthly_profit * 12 * np.minimum(1 / (1 - retention_rate), 10)  # Cap at 10 years
    
    def optimize(self, customer_ids, max_budget):
        
        # Generate all possible allocations
        candidates = []
        for cid in customer_ids:
            customer = self.data[self.data['CustomerId'] == cid].iloc[0]
            for strat, (cost, risk_red, _) in self.strategies.items():
                value_saved = customer['CLV'] * risk_red * customer['churn_prob']
                roi = min(value_saved - cost, customer['CLV'] * 0.9)  # ROI cap
                candidates.append({
                    'CustomerID': cid,
                    'Strategy': strat,
                    'ROI': roi,
                    'Cost': cost,
                    'Risk': customer['churn_prob'],
                    'CLV': customer['CLV']
                })
        
        # Sort by ROI efficiency
        candidates_df = pd.DataFrame(candidates)
        candidates_df['ROI_Ratio'] = candidates_df['ROI'] / candidates_df['Cost']
        candidates_df = candidates_df.sort_values('ROI_Ratio', ascending=False)
        
        # Greedy allocation
        allocations = []
        strategy_counts = {s:0 for s in self.strategies}
        remaining_budget = max_budget
        
        for _, row in candidates_df.iterrows():
            if remaining_budget < row['Cost']:
                continue
            if strategy_counts[row['Strategy']] >= self.strategies[row['Strategy']][2]:
                continue
                
            allocations.append(row.to_dict())
            strategy_counts[row['Strategy']] += 1
            remaining_budget -= row['Cost']
            
            if remaining_budget < min(v[0] for v in self.strategies.values()):
                break
        
        return {
            'allocations': pd.DataFrame(allocations),
            'strategy_counts': strategy_counts,
            'total_roi': sum(a['ROI'] for a in allocations),
            'budget_used': max_budget - remaining_budget
        }
