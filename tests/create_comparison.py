import pandas as pd
import json

# Load 15 candidates (1-year backtest results)
candidates_1y = pd.read_csv('data/top_15_validation_candidates.csv')

# Load final results (5-year backtest results)
with open('results/final_results.json', 'r', encoding='utf-8') as f:
    final_results = json.load(f)

# Extract 5-year results
results_5y = []
for risk_profile, data in final_results.items():
    config = data['strategy_configuration']
    perf = data['performance_summary']
    
    results_5y.append({
        'Risk_Profile': risk_profile,
        'Momentum_Period': config['Momentum Period'],
        'Value_Weight': config['Value Weight'],
        'Momentum_Weight': config['Momentum Weight'],
        'Low_Volatility_Weight': config['Low Volatility Weight'],
        'Top_N': config['Top N'],
        'Max_Weight': config['Max Weight'],
        'Total_Return_5Y': perf['Total Return'],
        'Annualized_Volatility_5Y': perf['Annualized Volatility'],
        'Annualized_Return_5Y': perf['Annualized Return'],
        'Sharpe_Ratio_5Y': perf['Sharpe Ratio']
    })

df_5y = pd.DataFrame(results_5y)

# Create a simple comparison table
print("=== BACKTEST COMPARISON: 1-YEAR vs 5-YEAR ===")
print(f"Total 15 candidates tested: {len(candidates_1y)}")
print(f"Final strategies selected: {len(df_5y)}")
print()

# Show 1-year results summary
print("--- 1-YEAR BACKTEST RESULTS (15 candidates) ---")
summary_1y = candidates_1y.groupby('Risk Profile').agg({
    'Sharpe Ratio': ['count', 'mean', 'max'],
    'Annualized Return': 'mean',
    'Annualized Volatility': 'mean'
}).round(3)
print(summary_1y)
print()

# Show 5-year results
print("--- 5-YEAR BACKTEST RESULTS (3 final strategies) ---")
for _, row in df_5y.iterrows():
    print(f"{row['Risk_Profile']}: Sharpe={row['Sharpe_Ratio_5Y']}, Return={row['Annualized_Return_5Y']}, Vol={row['Annualized_Volatility_5Y']}")
print()

# Save detailed comparison
candidates_1y['Backtest_Period'] = '1-Year'
df_5y_renamed = df_5y.rename(columns={
    'Risk_Profile': 'Risk Profile',
    'Momentum_Period': 'Momentum Period',
    'Value_Weight': 'Value Weight',
    'Momentum_Weight': 'Momentum Weight', 
    'Low_Volatility_Weight': 'Low Volatility Weight',
    'Top_N': 'Top N',
    'Max_Weight': 'Max Weight',
    'Total_Return_5Y': 'Total Return',
    'Annualized_Volatility_5Y': 'Annualized Volatility',
    'Annualized_Return_5Y': 'Annualized Return',
    'Sharpe_Ratio_5Y': 'Sharpe Ratio'
})
df_5y_renamed['Backtest_Period'] = '5-Year'

# Combine both datasets
combined_df = pd.concat([
    candidates_1y[['Risk Profile', 'Momentum Period', 'Value Weight', 'Momentum Weight', 
                   'Low Volatility Weight', 'Top N', 'Max Weight', 'Total Return', 
                   'Annualized Volatility', 'Annualized Return', 'Sharpe Ratio', 'Backtest_Period']],
    df_5y_renamed[['Risk Profile', 'Momentum Period', 'Value Weight', 'Momentum Weight',
                   'Low Volatility Weight', 'Top N', 'Max Weight', 'Total Return',
                   'Annualized Volatility', 'Annualized Return', 'Sharpe Ratio', 'Backtest_Period']]
], ignore_index=True)

# Save to CSV
combined_df.to_csv('data/backtest_comparison_1y_vs_5y.csv', index=False)
print("✅ Created data/backtest_comparison_1y_vs_5y.csv")